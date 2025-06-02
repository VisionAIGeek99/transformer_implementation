import numpy as np
import torch
from base import BaseTrainer

class Trainer(BaseTrainer):
    """
    Trainer class
    """
        
    def __init__(self, model, criterion, optimizer, config, device,
                 data_loader, valid_data_loader=None):
        super().__init__(model, criterion, optimizer, config)
        self.device = device
        self.data_loader = data_loader

    def _train_epoch(self, epoch: int):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return A log that contains average loss and metric in this epoch.
        """

        self.model.train()

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimzer.step()

            print(f"Train Epoch: {self.epochs}, Loss: {loss}")

            