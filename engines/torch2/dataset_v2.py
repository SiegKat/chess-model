# dataset_v2.py

from torch.utils.data import Dataset
import torch

class ChessDatasetV2(Dataset):
    """
    Updated dataset to handle (state, policy_target, value_target) tuples.
    """
    def __init__(self, X, y_policy, z_value):
        self.X = X
        self.y_policy = y_policy
        self.z_value = z_value

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to tensors on the fly
        return (
            torch.tensor(self.X[idx], dtype=torch.float32), 
            torch.tensor(self.y_policy[idx], dtype=torch.long),
            torch.tensor(self.z_value[idx], dtype=torch.float32)
        )
