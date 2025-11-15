import torch

class JEPADataset(torch.utils.data.Dataset):
    """Custom dataset for JEPA training that only requires CSI data (no labels)"""

    def __init__(self, data_x):
        self.data_x = torch.from_numpy(data_x).float()

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx]

