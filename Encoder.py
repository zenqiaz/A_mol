from typing import Tuple, Literal, Optional
import copy
import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean

from e3nn.o3 import Irreps, TensorProduct, FullyConnectedTensorProduct, Linear
from e3nn.nn import Gate, FullyConnectedNet

def irrep_diff(a: Irreps, b: Irreps) -> Irreps:
    """
    Return the Irreps that are in `a` but *not* in `b`,
    multiplicity-wise.

    Example
    -------
    >>> irrep_diff(Irreps("5x0e + 3x1o"), Irreps("2x0e"))
    Irreps('3x0e + 3x1o')
    """
    counter = {}
    for mul, ir in a:
        counter[ir] = counter.get(ir, 0) + mul
    for mul, ir in b:
        counter[ir] = counter.get(ir, 0) - mul
    diff_list = [(mul, ir) for ir, mul in counter.items() if mul > 0]
    return Irreps(diff_list).simplify()

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


class HeavyEncoderLayer_old(nn.Module):
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
        self.tp_msg = FullyConnectedTensorProduct(
            self.node_irreps_in,
            self.edge_attr_irreps,
            self.msg_irreps
        )

        # (2) gated non-linearity on aggregated nodes
        # split msg_irreps into scalar + gated parts
        if gate_scalars <= 0:
            raise ValueError("gate_scalars must be > 0")
        self.irreps_gate_scalar = Irreps(f"{gate_scalars}x0e")
        #self.irreps_gate_vec    = self.msg_irreps - self.irreps_gate_scalar
        self.irreps_gate_vec  = irrep_diff(self.msg_irreps, self.irreps_gate_scalar)
        self.lin_gate_in  = FullyConnectedTensorProduct(
            self.msg_irreps,
            self.node_irreps_in,
            self.irreps_gate_scalar + self.irreps_gate_vec
        )
        n_gates = sum(mul for mul, _ in self.irreps_gate_vec)      # e.g. 16 + 16 = 32
        irreps_gates = Irreps(f"{n_gates}x0e")  
        self.gate = Gate(
            self.irreps_gate_scalar,  [nn.Sigmoid()],
            irreps_gates,             [nn.Tanh()],
            self.irreps_gate_vec
        )
        #self.node_irreps_out = self.gate.irreps_out

        # (3) heavy self tensor-product
        self.heavy_tp = FullyConnectedTensorProduct(
            self.heavy_out_irreps,
            self.heavy_out_irreps,
            self.heavy_out_irreps
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
        node_msg = scatter_mean(msg, dst, dim=0, dim_size=x.size(0))

        # (2) Gate
        gate_in  = self.lin_gate_in(node_msg, torch.ones_like(node_msg[:, :1]))
        x_aggr   = self.gate(gate_in)             # (N, node_irreps_out.dim)

        # (3-a) merge duplicates → one tensor per heavy atom
        x_heavy  = to_heavy_merged(x_aggr, z, canonical, reduce=self.reduce)

        # (3-b) self TP on the heavy nodes
        x_heavy_tp = self.heavy_tp(x_heavy, x_heavy)  # (N_heavy, heavy_out_irreps.dim)

        # (3-c) broadcast back to all AO duplicates
        x_out = broadcast_heavy_back(x_aggr, x_heavy_tp, z, canonical)

        return x_out                               # Irreps = self.node_irreps_out




class HeavyEncoderLayer(nn.Module):
    r"""Edge-MLP-driven message → gate → heavy self-TP."""
    def __init__(
        self,
        node_irreps_in:   Irreps,
        edge_attr_irreps: Irreps,
        msg_irreps:       Irreps,
        *,
        radial_layers: int = 1,
        radial_neurons: int = 128,
        gate_scalars:   int = 16,
        heavy_out_irreps: Optional[Irreps] = None,
        reduce: str = "mean",                      
        num_neighbors: float = 3.0,
    ):
        super().__init__()
        self.reduce = reduce
        node_irreps_in   = Irreps(node_irreps_in).simplify()
        edge_attr_irreps = Irreps(edge_attr_irreps).simplify()
        msg_irreps       = Irreps(msg_irreps).simplify()

        # 1 ────────────────────────────────────────────────────────────
        # external-weight TP for messages
        instructions = []          # (i_in1, i_in2, i_out, mode, has_weight)
        irreps_mid   = []          # the actual (mul, irrep) pairs for the TP output

        for i, (mul_u, ir_u) in enumerate(node_irreps_in):          # u-copies
            for j, (mul_v, ir_v) in enumerate(edge_attr_irreps):
                prod = Irreps(ir_u * ir_v)                 # v-copies
                for k in range(len(msg_irreps)):             # CG: 1o⊗1o → 0e+1e+2e
                    mul_k, ir_k = msg_irreps[k]
                    if ir_k in prod:                        # keep only wanted outputs
                        #k = msg_irreps.index(ir_out)                        # next output slot
                        #irreps_mid.append((mul_u, ir_out))          # same multiplicity as input-1
                        instructions.append((i, j, k, "uvw", True))
        #print(instructions, node_irreps_in,
        #    edge_attr_irreps,
        #    msg_irreps,)
        self.tp_msg = TensorProduct(
            node_irreps_in,
            edge_attr_irreps,
            msg_irreps,
            instructions=instructions, 
            internal_weights=False,
            shared_weights=False,
        )
        self.edge_mlp = FullyConnectedNet(
            [edge_attr_irreps.dim] +
            radial_layers * [radial_neurons] +
            [self.tp_msg.weight_numel],
            torch.nn.functional.silu
        )
        # 2 ────────────────────────────────────────────────────────────
        #  Gate split
        self.irreps_gate_scalar = Irreps(f"{gate_scalars}x0e")
        self.irreps_gate_vec    = irrep_diff(msg_irreps, self.irreps_gate_scalar)
        n_gates = sum(mul for mul, _ in self.irreps_gate_vec)      # e.g. 16 + 16 = 32
        irreps_gates = Irreps(f"{n_gates}x0e")  

        self.lin_gate_in = FullyConnectedTensorProduct(
            msg_irreps, node_irreps_in,
            self.irreps_gate_scalar + irreps_gates + self.irreps_gate_vec,
        )
        self.gate = Gate(
            self.irreps_gate_scalar,  [nn.Sigmoid()],
            irreps_gates,             [nn.Tanh()],
            self.irreps_gate_vec
        )
        self.node_irreps_out = self.gate.irreps_out

        # 3 ────────────────────────────────────────────────────────────
        
        instructions = [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvv', True), (1, 0, 1, 'uvu', True), (1, 1, 0, 'uvw', True)]
        self.heavy_tp = TensorProduct(
            self.node_irreps_out,          # first  input  (x_heavy)
            self.node_irreps_out,          # second input  (x_heavy again)
            self.node_irreps_out,          # desired output irreps
            instructions=instructions,                    # ← only diagonal multiplicities
            internal_weights=True,         # learn weights inside the layer
            shared_weights=True,           # same filter for every copy
        )
        self.post_broadcast = Linear(
        self.node_irreps_out,          # in‑irreps  (e.g. 28‑dim)
        heavy_out_irreps               # out‑irreps (e.g. 48‑dim)
        )

    # ---------- helpers for heavy-atom merge -------------------------
    @staticmethod
    def _to_heavy(x_full, z, canonical, reduce="mean"):
        if z.shape[0] == x_full.shape[0]:               # per‑node
            heavy_mask = z > 1
        else:                                           # per‑atom → broadcast
            heavy_mask = z[canonical] > 1
        ids = canonical[heavy_mask]
        if reduce == "mean":
            return scatter_mean(x_full[heavy_mask], ids, dim=0,
                                dim_size=int(ids.max()) + 1)
        return scatter_add(x_full[heavy_mask], ids, dim=0,
                           dim_size=int(ids.max()) + 1)

    @staticmethod
    def _broadcast_heavy(x_full, x_heavy, z, canonical):
        y = x_full.clone()
        if z.shape[0] == x_full.shape[0]:               # per‑node
            heavy_mask = z > 1
        else:                                           # per‑atom → broadcast
            heavy_mask = z[canonical] > 1
        y[heavy_mask] = x_heavy[canonical[heavy_mask]]
        return y

    # ---------- forward ---------------------------------------------
    def forward(self, x, *, edge_index, edge_attr, z, canonical):
        src, dst = edge_index
        #x_attr = copy.deepcopy(x)
        # a) per-edge weights & message
        w = self.edge_mlp(edge_attr)                        # (E, W)
        #breakpoint()
        msg = self.tp_msg(x[src], edge_attr, w)             # (E, msg_dim)
        node_msg = scatter_add(msg, dst, dim=0,
                               dim_size=x.size(0))

        # b) gate
        gate_in = self.lin_gate_in(node_msg, x)
        x = self.gate(gate_in)

        # c) heavy-atom self TP
        x_heavy = self._to_heavy(x, z, canonical, self.reduce)
        x_heavy = self.heavy_tp(x_heavy, x_heavy)
        x = self._broadcast_heavy(x, x_heavy, z, canonical)
        x = self.post_broadcast(x)  

        return x                                            # (N, out_dim)