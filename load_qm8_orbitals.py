import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
from rdkit import Chem
import numpy as np


def radius_graph(pos, r_max, r_min, batch) -> torch.Tensor:
    # naive and inefficient version of torch_cluster.radius_graph
    r = torch.cdist(pos, pos)
    index = ((r < r_max) & (r > r_min)).nonzero().T
    index = index[:, batch[index[0]] == batch[index[1]]]
    return index

# ---------- helpers --------------------------------------------------------
def sdf_to_Z_pos(file, removeHs=False):
    """Return two lists:  [Z_i] and [pos_i]  for every mol in file."""
    Z_list, pos_list = [], []
    for m in Chem.SDMolSupplier(file, sanitize=False, removeHs=False):
        if m is None:                             # skip malformed blocks
            continue
        conf = m.GetConformer()
        Z    = torch.tensor(
                 [a.GetAtomicNum() for a in m.GetAtoms()],
                 dtype=torch.long)                # shape (N,)
        pos  = torch.tensor(
                 [conf.GetAtomPosition(i) for i in range(m.GetNumAtoms())],
                 dtype=torch.float32)             # shape (N,3)
        Z_list.append(Z)
        pos_list.append(pos)
    return Z_list, pos_list


def csv_to_labels(
        file: str,
        label_indices_plain: list[int],
        label_indices_log:  list[int],
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Parameters
    ----------
    file : str
        Path to the .csv file.
    label_indices_plain : list[int]
        Column indices to keep in **linear** scale.
    label_indices_log : list[int]
        Column indices to keep in **log** scale  (log(x + eps)).
    eps : float, optional
        Small constant to avoid log(0), default 1e-8.

    Returns
    -------
    torch.Tensor
        Shape (n_samples, len(label_indices_plain) + len(label_indices_log)),
        dtype float32.
    """
    df = pd.read_csv(file)

    # drop header row (index 0) and pull the requested columns
    plain_vals = df.iloc[:, label_indices_plain].to_numpy(dtype="float32")
    log_vals   = df.iloc[:, label_indices_log ].to_numpy(dtype="float32")

    # log-transform only the selected block
    log_vals = np.log(log_vals + eps, dtype="float32")

    # concatenate:  [ plain | log ]
    labels = np.concatenate([plain_vals, log_vals], axis=1)

    return torch.from_numpy(labels)                  # shape (n_mols, 4)


class MolPointsDataset(Dataset):
    def __init__(self, sdf_file, csv_file, mask_good, label_indices, label_indices_log):
        Z_all, pos_all = sdf_to_Z_pos(sdf_file)        # lists (len = N)
        y_all          = csv_to_labels(csv_file, label_indices, label_indices_log)       # (N, L) tensor
    
        # --- apply mask ONCE -------------------------------------------------
        self.z   = [F.one_hot(z, num_classes=10) for z, keep in zip(Z_all,   mask_good) if keep]
        self.pos = [p for p, keep in zip(pos_all, mask_good) if keep]
        self.y   = y_all[mask_good]                    # tensor slicing
    
    def __len__(self):
        return len(self.z)
    
    def __getitem__(self, idx):
        return {"z": self.z[idx], "pos": self.pos[idx], "y": self.y[idx]}

# ---------- the Dataset ----------------------------------------------------
#class MolPointsDataset(Dataset):
    """
    Each item is a dict with variable-length atom info + fixed-length label.

        sample = {
            "Z":   LongTensor (N,),    # atomic numbers
            "pos": FloatTensor (N,3),  # coordinates in Å
            "y":   FloatTensor (...),  # label vector for the molecule
        }
    """
    #
    #def __init__(self, sdf_file, csv_file):
    #    self.Z, self.pos = sdf_to_Z_pos(sdf_file)
    #    self.y           = csv_to_labels(csv_file)
    #    
    #    if len(self.Z) != len(self.y):
    #        raise ValueError(
    #            f"SDF has {len(self.Z)} mols but CSV has {len(self.y)} rows "
    #            "(after filtering)—make sure they align!"
    #        )
    #
    #def __len__(self):
    #    return len(self.Z)
   # 
   # def __getitem__(self, idx):
   #     return {"Z": self.Z[idx], "pos": self.pos[idx], "y": self.y[idx]}

#mask_good = torch.ones(16, dtype=torch.bool)
#bad_idx = [263, 273, 1596, 1601, 1621, 3937, 3939, 3940, 3948, 5628, 5855, 5876, 5887, 5978, 6472, 7086, 7093, 7128, 7168, 7328, 7374, 7377, 8113, 9694, 9724, 9946, 9947, 10046, 13500, 13778, 13975, 15169, 17456, 17457, 17955, 21058, 21059, 21256, 21259]
#mask_good[bad_idx] = False
#dataset = MolPointsDataset("example.sdf", "qm8_ex1.csv", mask_good)

def collate_flat_old(batch):
    """
    Accepts a list of samples
        {"Z": LongTensor (Ni,),        # atomic numbers
         "pos": FloatTensor (Ni,3),    # coordinates   (Å)
         "y":  FloatTensor (L,) }      # labels
    and returns a dict ready for e3nn 2101 models:
        "Z"    (N,)       long
        "pos"  (N,3)      float
        "batch"(N,)       long   – graph index for every atom
        "y"    (B,L)      float  – stacked labels
    """
    # ---- concatenate variable-length tensors ------------------------------
    pos   = torch.cat([item["pos"] for item in batch], dim=0)        # (N,3)
    z     = torch.cat([item["z"]   for item in batch], dim=0).float()    
    #z     = torch.cat([item["z"]   for item in batch], dim=0).unsqueeze(1).float()        # (N,)
    
    # ---- build batch vector ----------------------------------------------
    counts = torch.tensor([len(item["pos"]) for item in batch])      # (B,)
    batch_vec = torch.repeat_interleave(
        torch.arange(len(batch), device=pos.device), counts)         # (N,)
    
    # ---- stack labels -----------------------------------------------------
    y = torch.stack([item["y"] for item in batch])                   # (B,L)
    
    return {"pos": pos, "z": z, "batch": batch_vec, "y": y}

#loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_flat)


def collate_flat(batch):
    """
    Replicate each atom so that heavy atoms (Z > 1) appear 5-times
    and hydrogen-like atoms (Z == 1) appear once.

    Each input sample
        {"Z":  LongTensor (Ni,),        # atomic numbers
         "pos":FloatTensor (Ni,3),      # coordinates (Å)
         "y":  FloatTensor (L,)}

    Output (after replication)
        {"pos":   FloatTensor (N',3),
         "z":     LongTensor  (N',)     # atomic numbers (repeated)
         "batch": LongTensor  (N',)     # graph index per node
         "y":     FloatTensor (B,L)}
    """

    pos_chunks   = []      # list of (Ni', 3) tensors
    z_chunks     = []      # list of (Ni',)  tensors
    batch_chunks = []      # list of (Ni',)  tensors

    for g, item in enumerate(batch):
        Z  = item["Z"]         # (Ni,)
        R  = item["pos"]       # (Ni,3)

        # choose 5 for Z>1, else 1
        reps = torch.where(Z > 1, torch.tensor(5, device=Z.device),
                                      torch.tensor(1, device=Z.device))  # (Ni,)

        R_rep = torch.repeat_interleave(R,  reps, dim=0)   # (Ni',3)
        Z_rep = torch.repeat_interleave(Z,  reps, dim=0)   # (Ni',)

        pos_chunks.append(R_rep)
        z_chunks.append(Z_rep)
        batch_chunks.append(
            torch.full((R_rep.size(0),), g, dtype=torch.long, device=R.device)
        )

    # concatenate across graphs
    pos   = torch.cat(pos_chunks,   dim=0)   # (N',3)
    z     = torch.cat(z_chunks,     dim=0)   # (N',)
    batch_vec = torch.cat(batch_chunks, dim=0)  # (N',)

    # stack labels (unchanged)
    y = torch.stack([item["y"] for item in batch])  # (B,L)

    return {"pos": pos, "z": z, "batch": batch_vec, "y": y}


#batch = next(iter(loader))        # Python’s iterator protocol

if __name__ == "__main__":
    dataset = MolPointsDataset("qm8.sdf", "qm8.csv")

    def collate(batch):
        """Simple list-returning collate fn (no padding)."""
        return {
            k: [d[k] for d in batch]              # lists of Z, pos, y
            for k in batch[0]
        }

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate)

    for batch in loader:
        print(batch["Z"][0].shape, batch["pos"][0].shape, batch["y"][0].shape)
        break