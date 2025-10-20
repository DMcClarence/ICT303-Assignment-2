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
    train_size = int(len(dataset) * 0.7) # 70%
    valid_size = int(len(dataset) * 0.2) # 20%
    test_size = int(len(dataset) * 0.1)  # 10%
    training_data, validation_data, test_data = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=shuffle, num_workers=workers)
    valid_loader = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=shuffle, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, valid_loader, test_loader