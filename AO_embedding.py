import torch
from torch import nn
from typing import Tuple, Literal, Optional
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.data import Data, DataLoader, Batch
import numpy as np
from e3nn.o3 import Irreps, TensorProduct, spherical_harmonics
from e3nn.nn import Gate
from math import ceil

# 24 copies of scalars (ℓ=0, even)  +  8 copies of vectors (ℓ=1, odd)
AO_IRREPS = Irreps("24x0e + 8x1o")           # total dimensionality = 24·1 + 8·3 = 48
AO_DIM     = AO_IRREPS.dim                   # 48

def _atomic_number(symbol: str) -> int:
    return int(np.where(np.array(
        'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca'.split()
    ) == symbol)[0][0] + 1)
#   0  1  2  3  4

ORBITALS = ["1s", "2s", "2px", "2py", "2pz"]

# heavy atoms use all five; hydrogen uses only 1s (others map to a dummy index)
# ↓ Example – expand / reorder as you please
ATOM_TYPES = ["H", "C", "N", "O", "F"]

key_to_idx = {}
idx_to_key = []
for atom in ATOM_TYPES:
    for orb in ORBITALS:
        if atom == "H" and orb != "1s":
            continue                      # skip unused hydrogen orbitals
        key = (atom, orb)
        key_to_idx[key] = len(idx_to_key)
        idx_to_key.append(key)


TABLE_SIZE = len(idx_to_key)       

# ----------------------------------------------------------------------
def atom_to_ao_nodes(atom_symbol: str, xyz: torch.Tensor):
    """
    Returns two parallel Python lists:
        • node_idx : indices into the AOEmbedding table
        • pos      : xyz repeated for each AO node
    """
    if atom_symbol == "H":
        # Only the 1s orbital
        return [key_to_idx[("H", "1s")]], [xyz]
    
    # Heavy atom: five orbitals
    idx_list = [key_to_idx[(atom_symbol, orb)] for orb in ORBITALS]
    pos_list = [xyz] * len(ORBITALS)              # same coordinate for every AO
    return idx_list, pos_list


# ----------------------------------------------------------------------
def two_radius_graphs(
    pos: torch.Tensor,            # (N, 3)
    batch: torch.Tensor,          # (N,)
    *,
    r_max: float,
    eps_overlap: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return
        idx_no   : (2, E1) bidirectional edge index *without* overlaps
        idx_with : (2, E2) single-direction edge index *only* overlaps
    """
    d = torch.cdist(pos, pos, compute_mode = 'donot_use_mm_for_euclid_dist')                        # (N,N)
    same_mol = batch[:, None] == batch[None, :]      # mask intra-molecule
    N = d.size(0)
    
    # base mask: within cutoff, not self-loop, same molecule
    base = (d < r_max) & (~torch.eye(N, dtype=torch.bool, device=pos.device)) & same_mol
    
    # ------------------------------------------------------------------
    # Overlap edges (distance ≈ 0) – keep each pair only once (upper-tri)
    iu = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
    idx_with = (base & iu).nonzero(as_tuple=False).T
    # Non-overlap edges (distance > eps_overlap) – keep both directions
    idx_no = (base & (d > eps_overlap)).nonzero(as_tuple=False).T
    return idx_no, idx_with


def naive_radius_graph(pos, r_max, r_min, batch) -> torch.Tensor:
    """
    Quick, all-CPU fallback similar to torch_cluster.radius_graph.
    Returns a *bidirectional* edge list.
    """
    dist = torch.cdist(pos, pos)
    mask = (dist < r_max) & (dist > r_min)
    mask &= batch[:, None] == batch[None, :]
    edge_index = mask.nonzero(as_tuple=False).T       # (2,E)
    return edge_index
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 1. Molecule → AO graph(s) with / without overlaps
# ----------------------------------------------------------------------

def build_two_ao_graphs(atom_symbols,
                        coordinates,
                        cutoff: float       = 1.5,
                        eps_overlap: float  = 1.0e-6,
                        lmax: int           = 2,          # <= SH order
                        max_num_neighbors: int = 32):
    """
    Returns Data with:
        node_idx          : (N,)
        pos               : (N,3)
        edge_index_no     : (2,E1)  bidirectional, no overlaps
        edge_vec_no       : (E1,3)
        edge_sh_no        : (E1, (lmax+1)**2)  <-- NEW
        edge_index_with   : (2,E2)  single-dir, overlaps only
        edge_vec_with     : (E2,3)  (zeros)
    """
    # ---------- build nodes exactly as before --------------------------
    node_idx, pos = [], []
    for sym, xyz in zip(atom_symbols, coordinates):
        idx_chunk, pos_chunk = atom_to_ao_nodes(sym, torch.as_tensor(xyz, dtype=torch.float32))
        node_idx.extend(idx_chunk)
        pos.extend(pos_chunk)
    
    node_idx = torch.tensor(node_idx, dtype=torch.long)
    pos      = torch.stack(pos, dim=0)
    
    atom_batch = []
    atom_id = 0
    for sym in atom_symbols:
        n_orb = 5 if sym != "H" else 1
        atom_batch.extend([atom_id] * n_orb)
        atom_id += 1
    atom_batch = torch.tensor(atom_batch, dtype=torch.long)
    
    # ---------- edge construction -------------------------------------
    edge_no, edge_with = two_radius_graphs(
        pos, atom_batch, r_max=cutoff, eps_overlap=eps_overlap
    )
    
    edge_vec_no   = pos[edge_no[1]] - pos[edge_no[0]]              # (E1,3)
    edge_vec_with = torch.zeros(edge_with.size(1), 3, dtype=pos.dtype)
    
    # ---------- spherical-harmonic encoding ---------------------------
    #   sh(lmax, vec, normalize=True) returns irreps Yl up to `lmax`
    edge_sh_no = spherical_harmonics(lmax, edge_vec_no, normalize=True)             # (E1, (lmax+1)**2)
    
    return Data(node_idx=node_idx, pos=pos,
                edge_index_no=edge_no,   edge_vec_no=edge_vec_no,   edge_sh_no=edge_sh_no,
                edge_index_with=edge_with, edge_vec_with=edge_vec_with)




def build_two_ao_graphs_rdm1(
    atom_symbols,
    coordinates,
    rdm1: torch.Tensor,
    *,
    cutoff: float = 1.5,
    eps_overlap: float = 1.0e-6,
    lmax: int = 2,
):
    """
    Returns torch_geometric.data.Data with
        • node_idx, pos, canonical (as before)
        • edge_index_no / edge_attr_no
        • edge_index_with / edge_label_with   (1-RDM term)
        • z        ← derived here via `_atomic_number`
    """
    # ---------------- nodes -------------------------------------------
    node_idx, pos = [], []
    for sym, xyz in zip(atom_symbols, coordinates):
        idx, p = atom_to_ao_nodes(sym, torch.as_tensor(xyz, dtype=torch.float32))
        node_idx.extend(idx)
        pos.extend(p)
    
    node_idx = torch.tensor(node_idx, dtype=torch.long)
    pos = torch.stack(pos, dim=0)
    
    # ---------------- atom-level bookkeeping -------------------------
    
    atom_of_node = []           # AO‑index → atom‑index
    z_list       = []           # atomic numbers (one per atom)

    for atom_id, sym in enumerate(atom_symbols):
        z_val = _atomic_number(sym)
        z_list.append(z_val)

        n_orb = 5 if sym != "H" else 1
        atom_of_node.extend([atom_id] * n_orb)

    atom_of_node = torch.tensor(atom_of_node, dtype=torch.long)   # (N_nodes,)
    z_tensor     = torch.tensor(z_list,    dtype=torch.long)       # (n_atoms,)

    # -------------------------------- molecule‑level batch --------------
    # one configuration ⇒ one molecule ⇒ every node gets the same ID (0)
    mol_batch = torch.zeros_like(atom_of_node)                     # (N_nodes,)

    # -------------------------------- edge construction -----------------
    edge_no, edge_with = two_radius_graphs(
        pos,
        batch       = mol_batch,       # ← fixed
        r_max       = cutoff,
        eps_overlap = eps_overlap,
    )
    
    edge_vec_no = pos[edge_no[0]] - pos[edge_no[1]]
    edge_attr_no = spherical_harmonics(
        list(range(lmax + 1)), edge_vec_no,
        normalize=False, normalization="component",
    )
    
    # 1-RDM label on overlap edges
    src_a = edge_with[0]
    dst_a = edge_with[1]
    edge_label_with = rdm1[src_a, dst_a]
    
    return Data(
        node_idx=node_idx,
        pos=pos,
        canonical=atom_of_node,        # map AO → parent atom
        z=z_tensor,                    # atomic numbers now included
        # edges without overlap
        edge_index_no=edge_no,
        edge_attr_no=edge_attr_no,
        # overlap edges with 1-RDM label
        edge_index_with=edge_with,
        edge_label_with=edge_label_with,
    )

# heavy_encoder.py
# ---------------------------------------------------------------------
#  E(3)-equivariant encoder layer with three blocks:
#   (1)  edge convolution  (node × edge_attr  →  message → aggregate)
#   (2)  gated update      (Gate on the aggregated node tensor)
#   (3)  heavy-atom self-TP (merge heavy duplicates → TP → broadcast back)
#
#  Author:  <you>
# ---------------------------------------------------------------------


# ------------------------------------------------------------
#  helper: heavy merge & broadcast (from your previous snippet)
# ------------------------------------------------------------
def to_heavy_merged(
    x_full: torch.Tensor,
    z: torch.Tensor,
    canonical: torch.Tensor,
    *,
    reduce: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    heavy_mask = z > 1
    can_heavy  = canonical[heavy_mask]
    num_heavy  = int(can_heavy.max()) + 1

    if reduce == "mean":
        return scatter_mean(x_full[heavy_mask], can_heavy, dim=0, dim_size=num_heavy)
    elif reduce == "sum":
        return scatter_add(x_full[heavy_mask],  can_heavy, dim=0, dim_size=num_heavy)
    else:
        raise ValueError

def broadcast_heavy_back(
    x_full: torch.Tensor,
    x_heavy: torch.Tensor,
    z: torch.Tensor,
    canonical: torch.Tensor,
) -> torch.Tensor:
    heavy_mask  = z > 1
    x_full_upd  = x_full.clone()
    x_full_upd[heavy_mask] = x_heavy[canonical[heavy_mask]]
    return x_full_upd
# ------------------------------------------------------------

class HeavyEncoderLayer(nn.Module):
    r"""
    One equivariant “encoder” block, matching your three–arrow sketch:

        1. Convolution
        2. Gate
        3. Heavy-atom self tensor-product

    Parameters
    ----------
    node_irreps_in    : Irreps of input node features
    edge_attr_irreps  : Irreps carried by each edge (e.g. SH up to l_max)
    msg_irreps        : Irreps produced by the edge tensor-product
    gate_scalars      : Number of even scalars driving the Gate
    heavy_out_irreps  : Irreps produced by the heavy self-TP
    reduce            : “mean” or “sum” when merging AO duplicates
    """

    def __init__(
        self,
        node_irreps_in:   Irreps,
        edge_attr_irreps: Irreps,
        msg_irreps:       Irreps,
        *,
        gate_scalars: int = 16,
        heavy_out_irreps: Optional[Irreps] = None,
        reduce: Literal["mean", "sum"] = "mean",
    ):
        super().__init__()
        self.node_irreps_in   = Irreps(node_irreps_in).simplify()
        self.edge_attr_irreps = Irreps(edge_attr_irreps).simplify()
        self.msg_irreps       = Irreps(msg_irreps).simplify()
        self.reduce           = reduce

        # (1) edge convolution:   node ⊗ edge_attr → message
        self.tp_msg = TensorProduct(
            self.node_irreps_in,
            self.edge_attr_irreps,
            self.msg_irreps,
            shared_weights=False,
            internal_weights=False,
        )

        # (2) gated non-linearity on aggregated nodes
        # split msg_irreps into scalar + gated parts
        if gate_scalars <= 0:
            raise ValueError("gate_scalars must be > 0")
        self.irreps_gate_scalar = Irreps(f"{gate_scalars}x0e")
        self.irreps_gate_vec    = self.msg_irreps - self.irreps_gate_scalar
        self.lin_gate_in  = TensorProduct(
            self.msg_irreps,
            Irreps("0e"),
            self.irreps_gate_scalar + self.irreps_gate_vec,
            shared_weights=True,
            internal_weights=False,
        )
        self.gate = Gate(
            self.irreps_gate_scalar,  [nn.Sigmoid()],
            self.irreps_gate_vec,     [nn.Tanh()],
        )
        self.node_irreps_out = self.gate.irreps_out

        # (3) heavy self tensor-product
        self.heavy_tp = TensorProduct(
            self.node_irreps_out,
            self.node_irreps_out,
            heavy_out_irreps or self.node_irreps_out,
            shared_weights=False,
            internal_weights=False,
        )

    # --------------------------------------------------------
    #  forward
    # --------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                  # (N, node_irreps_in.dim)
        edge_index: torch.Tensor,         # (2, E)  src → dst
        edge_attr: torch.Tensor,          # (E, edge_attr_irreps.dim)
        z: torch.Tensor,                  # (N,)  atomic numbers
        canonical: torch.Tensor,          # (N,)  duplicate-to-heavy map
    ) -> torch.Tensor:
        """Return updated node features with heavy-atom mixing applied."""
        src, dst = edge_index                     # unpack

        # (1) message = TP(node_src, edge_attr)  and aggregate at dst
        msg  = self.tp_msg(x[src], edge_attr)     # (E, msg_irreps.dim)
        node_msg = scatter_add(msg, dst, dim=0, dim_size=x.size(0))

        # (2) Gate
        gate_in  = self.lin_gate_in(node_msg, torch.ones_like(node_msg[:, :1]))
        x_aggr   = self.gate(gate_in)             # (N, node_irreps_out.dim)

        # (3-a) merge duplicates → one tensor per heavy atom
        x_heavy  = to_heavy_merged(x_aggr, z, canonical, reduce=self.reduce)

        # (3-b) self TP on the heavy nodes
        x_heavy_tp = self.heavy_tp(x_heavy, x_heavy)  # (N_heavy, heavy_out_irreps.dim)

        # (3-c) broadcast back to all AO duplicates
        x_out = broadcast_heavy_back(x_aggr, x_heavy_tp, z, canonical)

        return x_out                               # Irreps = heavy_tp.irreps_out



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




def build_ao_graphs_labels(
    atom_symbols,
    coordinates,
    rdm1: torch.Tensor,
    *,
    cutoff: float = 1.5,
    eps_overlap: float = 1.0e-6,
    lmax: int = 2,
    graph_label: torch.tensor
):
    """
    Build a PyG Data object and attach ONE scalar label `graph_y`.
      • node_idx, pos, canonical, z
      • edge_index_no / edge_attr_no     (for message passing)
      • edge_index_with                  (overlap edges; no label stored)
      • graph_y          ← scalar derived from the full 1-RDM
    """
    # ----- nodes -----------------------------------------------------
    node_idx, pos = [], []
    for sym, xyz in zip(atom_symbols, coordinates):
        idx, p = atom_to_ao_nodes(sym, torch.as_tensor(xyz, dtype=torch.float32))
        node_idx.extend(idx)
        pos.extend(p)

    node_idx = torch.tensor(node_idx, dtype=torch.long)
    pos      = torch.stack(pos, dim=0)

    # ----- atom bookkeeping -----------------------------------------
    atom_of_node = []
    z_list       = []
    for atom_id, sym in enumerate(atom_symbols):
        z_list.append(_atomic_number(sym))
        atom_of_node.extend([atom_id] * (5 if sym != "H" else 1))

    atom_of_node = torch.tensor(atom_of_node, dtype=torch.long)  # (N_nodes,)
    z_tensor     = torch.tensor(z_list,     dtype=torch.long)    # (n_atoms,)

    # ----- single-molecule batch vector -----------------------------
    mol_batch = torch.zeros_like(atom_of_node)                   # (N_nodes,)

    # ----- edge construction (same as before) -----------------------
    edge_no, edge_with = two_radius_graphs(
        pos, mol_batch, r_max=cutoff, eps_overlap=eps_overlap)

    edge_vec_no  = pos[edge_no[0]] - pos[edge_no[1]]
    edge_attr_no = spherical_harmonics(
        list(range(lmax + 1)), edge_vec_no,
        normalize=False, normalization="component",
    )

    # ----------------------------------------------------------------
    # graph-level label: choose your formula here
    # ----------------------------------------------------------------
    graph_y = torch.tensor([float(graph_label)], dtype=torch.float32)

    # ----- package ---------------------------------------------------
    return Data(
        node_idx=node_idx,
        pos=pos,
        canonical=atom_of_node,
        z=z_tensor,
        edge_index_no=edge_no,
        edge_attr_no=edge_attr_no,
        edge_index_with=edge_with,   # still used by encoder; no label here
        graph_y=graph_y,             # <── NEW scalar target (shape 1,)
    )
# ------------------------------------------------------------------
# 3.   Quick smoke test -------------------------------------------------
if __name__ == "__main__":
    # minimal fake graph (3 atoms → 7 AO nodes)
    from e3nn import o3
    Ntok = len(idx_to_key)                 # dictionary built earlier
    EDGE_IRREPS = Irreps("0e + 1o + 2e")   # l_max = 2
    MSG_IRREPS  = Irreps("16x0e + 8x1o")

    model = PretrainEncoder(
        num_tokens=Ntok,
        edge_attr_irreps=EDGE_IRREPS,
        msg_irreps=MSG_IRREPS,
        n_layers=2,
    )

    # tensor sizes ------------------------------------------------------
    g = build_two_ao_graphs(              # ← from previous helper
        ["O", "H", "H"],
        [[0., 0., 0.],
         [0.7586, 0., 0.5043],
         [-0.7586, 0., 0.5043]],
        lmax=2
    )

    out = model(g)                        # (N_nodes, 1)
    print("OK – output shape:", out.shape)


