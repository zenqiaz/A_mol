import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_add
import math
from typing import Dict, Tuple, List, Optional
import numpy as np            # only for tiny periodic-table helper
from AO_embedding import build_two_ao_graphs_rdm1, idx_to_key
from Encoder import HeavyEncoderLayer
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader, Batch, Data

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, TensorProduct
from e3nn.util.jit import compile_mode


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)

def widen(ir: Irreps, factor: int) -> Irreps:
    """
    Return a copy where each term's *multiplicity* is multiplied by `factor`.
    widen("24x0e + 8x1o", 2)  →  "48x0e + 16x1o"
    """
    return Irreps([
        (mul * factor, irrep) for mul, irrep in ir
    ])

def radius_graph(pos, r_max, r_min, batch) -> torch.Tensor:
    # naive and inefficient version of torch_cluster.radius_graph
    r = torch.cdist(pos, pos)
    index = ((r < r_max) & (r > r_min)).nonzero().T
    index = index[:, batch[index[0]] == batch[index[1]]]
    return index

def _atomic_number(symbol: str) -> int:
    return int(np.where(np.array(
        'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca'.split()
    ) == symbol)[0][0] + 1)
from typing import Tuple, Literal

BASE_IRREPS = Irreps("24x0e + 8x1o")         # 48-dim input rows

#layer_width = [1, 2, 4, 4]                   # multiplicative factors
# ------------------------------------------------------------
#  helper: heavy merge & broadcast (from your previous snippet)
# ------------------------------------------------------------

# ------------------------------------------------------------------
# 1.   Token → equivariant row  (unchanged, just imported)
# ------------------------------------------------------------------
class AOEmbedding(nn.Module):
    """Lookup table whose rows carry Irreps."""
    IRREPS = Irreps("24x0e + 8x1o")          # <-- keep in one place

    def __init__(self, num_tokens: int, freeze_s_vectors: bool = True):
        super().__init__()
        self.weight = nn.Parameter(
            0.1 * torch.randn(num_tokens, self.IRREPS.dim)
        )

        # optional mask: zero–and–freeze ℓ=1 part of every s-orbital row
        if freeze_s_vectors:
            mask = torch.ones_like(self.weight)
            # assume 24×0e come first → vectors start at index 24
            start_vec = 24
            _is_s = torch.tensor([idx_to_key[i][1].endswith("s")
                                  for i in range(num_tokens)])
            mask[_is_s, start_vec:] = 0.0
            self.register_buffer("mask", mask, persistent=False)
        else:
            self.mask = None

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        w = self.weight[idx]
        if self.mask is not None:
            w = w * self.mask[idx]      # gradients stay 0 on masked entries
        return w                        # (N, 48) – carries Irreps



class PretrainEncoder(nn.Module):
    r"""
    End-to-end encoder

        one-hot index  →  AOEmbedding  →  N×HeavyEncoderLayer
    """
    def __init__(
        self,
        *,
        num_tokens: int,
        node_irreps_ins: list,
        edge_attr_irreps: Irreps,
        msg_irreps: Irreps,
        n_layers: int = 4,
        gate_scalars: int = 16,
        l_heavy: int = 2,           # self-TP l-max (controls heavy_out_irreps)
    ):
        super().__init__()

        # 2-a  Input embedding
        self.embed = AOEmbedding(num_tokens)

        # 2-b  A stack of HeavyEncoderLayers (all share the same edge irreps)
        heavy_out = Irreps.spherical_harmonics(l_heavy, p=0)  # 0e+1o+... up to l_heavy
        layers = []
        for layer_index in range(n_layers):
            layer = HeavyEncoderLayer(
                node_irreps_in=node_irreps_ins[layer_index],
                edge_attr=edge_attr_irreps,
                msg_irreps=msg_irreps,
                gate_scalars=gate_scalars,
                heavy_out_irreps=heavy_out,
            )
            layers.append(layer)
            node_irreps_in = layer.heavy_tp.irreps_out      # chain output→input
        self.layers = nn.ModuleList(layers)

        # 2-c  (optional) small head for a self-supervised target --------------
        self.head = nn.Linear(node_irreps_in.dim, 1)        # e.g. scalar energy

    # ------------------------------------------------------------------
    def forward(self, data):
        """
        Expected `data` fields:
            • node_idx           (N,)   int
            • edge_index_no      (2,E)
            • edge_attr_no       (E, edge_attr_irreps.dim)
            • z, canonical       (N,)   tensors for heavy merge/broadcast
        """
        x = self.embed(data.node_idx)                       # 48-d equivariant

        for layer in self.layers:
            x = layer(
                x            = x,
                edge_index   = data.edge_index_no,
                edge_attr    = data.edge_attr_no,
                z            = data.z,
                canonical    = data.canonical,
            )

        return self.head(x)                                 # (N,1) or adapt



# ---------------- 1. Encoder with explicit width schedule ---------------------
class Encoder(nn.Module):
    r"""
    Equivariant encoder stack.

    Parameters
    ----------
    base_irreps        : Irreps of the *initial* node embedding
    width_factors      : list[int]; multiplicative width for each layer
    edge_attr_irreps   : Irreps of edge attributes (e.g. SH)
    msg_irreps_base    : template Irreps for message TP (will be widened)
    gate_scalars       : # of 0e channels driving the Gate
    """

    def __init__(
        self,
        base_irreps: Irreps,
        width_factors: List[int],
        edge_attr_irreps: Irreps,
        msg_irreps_base: Irreps,
        gate_scalars: int = 16,
        num_neighbors: float = 3.0
    ):
        super().__init__()
        self.base_irreps   = Irreps(base_irreps)
        self.width_factors = width_factors               # ← attribute you asked for

        layers   = nn.ModuleList()
        in_ir    = self.base_irreps

        for w in width_factors:
            out_ir = widen(self.base_irreps, w)
            msg_ir = widen(msg_irreps_base, w)

            layer = HeavyEncoderLayer(
                node_irreps_in   = in_ir,
                edge_attr_irreps = edge_attr_irreps,
                msg_irreps       = msg_ir,
                gate_scalars     = gate_scalars,
                heavy_out_irreps = out_ir,
                num_neighbors = num_neighbors
            )
            layers.append(layer)
            in_ir = out_ir                              # next layer sees widened irs

        self.layers      = layers
        self.out_irreps  = in_ir                        # keep for decoder

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x0                                     # node embedding already done
        for lyr in self.layers:
            x = lyr(x,
                     edge_index = data.edge_index_no,
                     edge_attr  = data.edge_attr_no,
                     z          = data.z,
                     canonical  = data.canonical)
        return x                                        # (N_nodes, out_irreps.dim)


# ---------------- 2. Edge decoder --------------------------------------------
class EdgeDecoder(nn.Module):#decoder for edge label output
    """
    Predict one scalar per *edge* in `edge_index_with`.
    - Keeps equivariance by first dropping vectors/tensors → scalars (0e).
    """

    def __init__(self, node_irreps: Irreps, scalar_channels: int = 64):
        super().__init__()
        edge_scalar_irreps = Irreps(f"{scalar_channels}x0e")  
        # 2-a  reduce each node feature to purely scalar channels
        self.edge_tp = FullyConnectedTensorProduct(
        irreps_in1=node_irreps,
        irreps_in2=node_irreps,
        irreps_out=edge_scalar_irreps,
        # -------- weight handling ----------
        #internal_weights=True,   # θ lives *inside* the module
        #shared_weights=True,     # the same θ is reused for every call / every edge
        # (mode defaults to "uvu", which is fine for copy‑by‑copy sharing)
)

        # 2-b  tiny MLP on concatenated [src || dst] scalars
        self.mlp = nn.Sequential(
            nn.Linear(scalar_channels, scalar_channels//2),
            nn.SiLU(),
            nn.Linear(scalar_channels//2, 1)
        )

    def forward(self, node_tensors, edge_index_with) -> torch.Tensor:
        src, dst = edge_index_with           # (N, C)
        e = self.edge_tp(node_tensors[src], node_tensors[dst])        # (E, 2C)
        #breakpoint()
        return self.mlp(e).squeeze(-1)                 # (E,)



class GraphDecoder_gate(nn.Module):
    """
    Node → scalar-gate → attention-pooled graph vector → MLP → graph label.
    """
    def __init__(self, node_irreps, hidden=64, out_dim=1):
        super().__init__()

        # 1. equivariant node → hidden 0e scalars
        self.to_scalar = e3nn_nn.Linear(node_irreps, f"{hidden}x0e")

        # 2. gating network that produces one weight α_i per node
        gate_nn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1)        # α_i (pre-softmax)
        )
        self.pool = GlobalAttention(gate_nn)  # learns attention weights

        # 3. final MLP on pooled graph representation
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim)        # graph label(s)
        )

class GraphDecoder(nn.Module):#decoder for graph label output
    """
    Pool node features → predict one scalar (or small vector) per graph.
    """
    def __init__(self, node_irreps, hidden=64, out_dim=1):
        super().__init__()
        # turn equivariant node tensor into hidden scalars
        self.to_scalar = e3nn_nn.Linear(node_irreps, f"{hidden}x0e")

        # simple MLP on pooled graph representation
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim)      # graph label(s)
        )

    def forward(self, node_feat, batch_vec):
        # 1. per-node → scalars
        h_node = self.to_scalar(node_feat)       # (N, hidden)

        # 2. pool to graph (mean works well for extensive/intensive labels)
        h_graph = global_mean_pool(h_node, batch_vec)   # (G, hidden)

        # 3. predict graph label
        return self.mlp(h_graph).squeeze(-1)     # (G,) if out_dim = 1



class Model_edge(nn.Module):
    """
    AOEmbedding  →  Encoder  →  EdgeDecoder
    """

    def __init__(
        self,
        *,
        num_tokens: int,
        base_irreps: Irreps,
        width_factors: List[int],
        edge_attr_irreps: Irreps,
        msg_irreps_base: Irreps,
        gate_scalars: int = 16,
        scalar_channels: int = 64,
    ):
        super().__init__()

        # --- 0. token → equivariant row ------------------------------
        self.embed = AOEmbedding(num_tokens)

        # --- 1. equivariant stack ------------------------------------
        self.encoder = Encoder(
            base_irreps       = base_irreps,
            width_factors     = width_factors,
            edge_attr_irreps  = edge_attr_irreps,
            msg_irreps_base   = msg_irreps_base,
            gate_scalars      = gate_scalars,
        )

        # --- 2. edge‑level decoder -----------------------------------
        self.graph_dec = GraphDecoder(
            node_irreps = self.encoder.out_irreps,
            hidden      = graph_hidden,
            out_dim     = 1                 # one scalar per molecule
        )

    def forward(self, data: Data) -> torch.Tensor:
        # 0 → build node features on the fly
        data.x0 = self.embed(data.node_idx)           # (N, base_irreps.dim)

        # 1 → encode
        node_feat = self.encoder(data)                # (N, out_dim)

        # 2 → edge prediction for overlap edges
        graph_pred = self.graph_dec(node_feat)
        return graph_pred    


class Model_Graph(nn.Module):
    """
    AOEmbedding  →  Encoder  →  EdgeDecoder
    """

    def __init__(
        self,
        *,
        num_tokens: int,
        base_irreps: Irreps,
        width_factors: List[int],
        edge_attr_irreps: Irreps,
        msg_irreps_base: Irreps,
        gate_scalars: int = 16,
        scalar_channels: int = 64,
    ):
        super().__init__()

        # --- 0. token → equivariant row ------------------------------
        self.embed = AOEmbedding(num_tokens)

        # --- 1. equivariant stack ------------------------------------
        self.encoder = Encoder(
            base_irreps       = base_irreps,
            width_factors     = width_factors,
            edge_attr_irreps  = edge_attr_irreps,
            msg_irreps_base   = msg_irreps_base,
            gate_scalars      = gate_scalars,
        )

        # --- 2. edge‑level decoder -----------------------------------
        self.decoder = EdgeDecoder(
            node_irreps       = self.encoder.out_irreps,
            scalar_channels   = scalar_channels,
        )

    def forward(self, data: Data) -> torch.Tensor:
        # 0 → build node features on the fly
        data.x0 = self.embed(data.node_idx)           # (N, base_irreps.dim)

        # 1 → encode
        node_feat = self.encoder(data)                # (N, out_dim)

        # 2 → edge prediction for overlap edges
        pred = self.decoder(node_feat, data.edge_index_with)  # (E2,)
        return pred



class RDM1Dataset(Dataset):
    def __init__(self, raw_items, lmax=2):
        # raw_items is already a *list*, so len() works
        self.items = raw_items
        self.lmax  = lmax

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        config, (_, rdm1) = self.items[idx]     # <- notice (_, rdm1)

        symbols, coords  = zip(*config)         # split geometry
        data = build_two_ao_graphs_rdm1(
            list(symbols), list(coords), torch.as_tensor(rdm1, dtype=torch.float32), lmax=self.lmax
        )
        return data


class LabelDataset(Dataset):
    def __init__(self, raw_items, lmax=2):
        # raw_items is already a *list*, so len() works
        self.items = raw_items
        self.lmax  = lmax

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        z, pos, y = self.items[idx]     # <- notice (_, rdm1)

        symbols, coords  = zip(*config)         # split geometry
        data = build_ao_graphs_labels(list(z), list(pos), torch.as_tensor(y, dtype=torch.float32), lmax=self.lmax)
        return data

