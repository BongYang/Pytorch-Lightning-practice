import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from torchmetrics import functional as FM
from torchvision.datasets import MNIST


class MNISTClassifier(LightningModule):
    def __init__(self, cfg):
        super(MNISTClassifier, self).__init__()

        self.batch_size = cfg.batch_size
        self.lr = cfg.lr
        self.cpus = cfg.cpus
        self.image_shape = (1, 28, 28)
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            torch.nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)
        
    def prepare_data(self):
        transform=transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5])
            ])
        
        self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, self.batch_size, num_workers=self.cpus)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_test, self.batch_size, num_workers=self.cpus)
        
    def test_dataloader(self):
        return DataLoader(self.mnist_test, self.batch_size, num_workers=self.cpus)
        
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        
        return {'loss': loss}
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self.forward(x)
        loss = self.cross_entropy_loss(pred, y)
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        acc = FM.accuracy(pred, y)
        loss = F.cross_entropy(pred, y)
        return {'test_acc': acc, 'test_loss': loss}
    
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_acc': avg_acc, 'test_loss': avg_loss}
        return {'avg_test_acc': avg_acc, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
