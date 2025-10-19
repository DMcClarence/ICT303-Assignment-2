# Data Loader

import torch
from torchvision import datasets, transforms

class DataLoader:
  def __init__(self, data_dir, trans_width, trans_height,):
    self.width = trans_width
    self.height = trans_height
    self.root = data_dir

  def load(self, transform, batch_size=10, shuffle=True, workers=2):
    dataset = datasets.ImageFolder(root=self.root, transform=transform)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    return dataset