import torch
import wandb
from abc import abstractmethod

class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, optimizer, config):
        self.config = config
        self.wandb = None

        self.model = model
        self.optimzer = optimizer
        self.criterion = criterion
        
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.start_epoch = 1
        

    def _init_wandb(self):
        wandb.init(project=self.config["project_name"], config=self.config)
        self.wandb = wandb

    def log_metrics(self, metrics, step=None):
        self.wandb.log(metrics, step=step)

    def finish_wandb(self):
        self.wandb.finish()

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    def train(self):
        """
        Full training logic
        """

        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)



    def _save_checkpoint(self, epoch, save_base=False):
        pass

    def _resume_checkpoint(self, resume_path):
        pass

    