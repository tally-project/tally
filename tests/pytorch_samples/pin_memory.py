import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __len__(self):
        return 1000  # Example size of the dataset

    def __getitem__(self, idx):
        data = torch.randn(3, 224, 224)  # Example data (random tensor)
        label = torch.randint(0, 10, (1,))  # Example label
        return data, label
    
dataset = CustomDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

for data, labels in loader:
    data, labels = data.to('cuda', non_blocking=True), labels.to('cuda', non_blocking=True)
    # Your training logic goes here