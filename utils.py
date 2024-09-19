import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopper:
    def __init__(self, patience=15, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf


def set_optimizer(cfg, model):
    optimizer = optim.Adam([paras for paras in model.parameters() if paras.requires_grad == True], 
                            lr=cfg.lr, 
                            amsgrad=True)

    return optimizer

def save_model(model, optimizer, cfg, epoch, save_file):
    print('==> Saving...')
    state = {
        'cfg': cfg,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


class SaveModel:
    def __init__(self, cfg, save_file):
        self.cfg = cfg
        self.save_file = save_file
        if cfg.task == 'Classification':
            self.best_metric = 0
        elif cfg.task == 'Tree':
            self.best_metric = 0
        else:
            self.best_metric = 1000

    def save(self, model, optimizer, scheduler, metric):
        if self.cfg.task == 'Classification' or self.cfg.task == 'Tree' or self.cfg.task =='H':
            if metric > self.best_metric:
                self.best_metric = metric 
                print('=====> Best Model Saving (Acc) <=====')
                state = {
                    'cfg': self.cfg,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                }
                torch.save(state, self.save_file)
                del state
        else:
            if metric < self.best_metric:
                self.best_metric = metric 
                print('==> Best Model Saving (Loss) ...')
                state = {
                    'cfg': self.cfg,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                }
                torch.save(state, self.save_file)
                del state