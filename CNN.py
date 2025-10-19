import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class CNN(nn.Module):
    def __init__(self, log_dir, lr=0.0001, outputs=2):
        super(CNN, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(512, outputs),
        )

        self.lr = lr
        self.writer = SummaryWriter(log_dir=log_dir)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def loss(y_hat, y):
        fn = nn.CrossEntropyLoss()
        return fn(y_hat, y)

    def configure_optimiser(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def save(self, save_dir, trained_epochs=0):
        save_path = (save_dir + f"/CNN_Epoch_{int(trained_epochs)}.tar")
        torch.save(
            dict(model=self.net.state_dict(),
                 learning_rate=self.lr,
                 epochs_trained=trained_epochs),
            save_path)
        print(f"CNN saved to {save_path} at Epoch {trained_epochs}")

    def load(self, load_dir):
        checkpoint = torch.load(load_dir, weights_only=True)
        self.model.load_state_dict(checkpoint['model'])
        self.lr = checkpoint['learning_rate']
        epochs = checkpoint['epochs_trained']
        return epochs