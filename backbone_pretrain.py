import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import EarlyStopper
from models.resnets import parse_model_from_name
import dataset

# Set seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_folder', type=str, default='ckpt')
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--save', type=bool, default=True, help='Flag to save the model weights')
    parser.add_argument("--model", type=str, default='euclidean_without_avgpool', choices= ["euclidean", "euclidean_without_avgpool", "euclidean_masking"], help="Model name")
    parser.add_argument("--settings", type=str, default='fft_RandomCrop', choices=['fft_RandomCrop', 'fft_FixedCrop', 'govdocs_RandomCrop'], help="Select settings")

    return parser.parse_args()


def set_loader(args):
    global train
    global valid
    print(f'====================== {args.settings} ======================')

    if args.settings == 'fft_RandomCrop':
        from trainer import backbone_train as train
        from trainer import backbone_valid as valid

        train_data = dataset.BackboneDataset(dataset_type='fft', mode='train', is_fixed=False)
        valid_data = dataset.BackboneDataset(dataset_type='fft', mode='val', is_fixed=False)
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)

        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        
        ckpt_path = 'FFT75_RandomCrop_ResNet18.pth'

        return train_loader, valid_loader, cnn_backbone, ckpt_path

    if args.settings == 'fft_FixedCrop':
        from trainer import backbone_train as train
        from trainer import backbone_valid as valid

        train_data = dataset.BackboneDataset(dataset_type='fft', mode='train', is_fixed=True)
        valid_data = dataset.BackboneDataset(dataset_type='fft', mode='val', is_fixed=True)
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)

        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        
        ckpt_path = 'FFT75_FixedCrop_ResNet18.pth'
        
        return train_loader, valid_loader, cnn_backbone, ckpt_path


    elif args.settings == 'govdocs_RandomCrop':
        from trainer import backbone_train as train
        from trainer import backbone_valid as valid

        train_data = dataset.BackboneDataset(dataset_type='govdocs', mode='train', is_fixed=True)
        valid_data = dataset.BackboneDataset(dataset_type='govdocs', mode='val', is_fixed=True)
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)

        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        
        ckpt_path = 'Govdocs_RandomCrop_ResNet18.pth'

        return train_loader, valid_loader, cnn_backbone, ckpt_path


def main():
    args = parse_args()     

    # Create some strings for file management
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(dir_path, args.save_folder)
    print(f"Experiment directory: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)

    train_loader, valid_loader, cnn_backbone, ckpt_path = set_loader(args)

    # Initialize optimizer
    optimizer = torch.optim.Adam(cnn_backbone.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
    early_stopper = EarlyStopper(patience=11, verbose=True)

    print(f"Using optimizer: {optimizer}")

    criterion = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train for one epoch
        train_loss = train(args, train_loader, cnn_backbone, criterion, optimizer)

        # Validate for one epoch
        valid_loss, valid_acc = valid(args, valid_loader, cnn_backbone, scheduler, early_stopper)
        print(f'valid_loss : {valid_loss}, valid_acc : {valid_acc}')
        # Log metrics to wandb

        # Early stopping check
        print(f"Calling early_stopper for epoch {epoch}") 
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Save the best model if valid_loss decreases
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = {
                'cnn': cnn_backbone.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            if args.save:
                save_path = os.path.join(exp_dir, f"/backbone/{ckpt_path}.pth")
                torch.save(best_model_state, save_path)
                print(f"Model saved at epoch {epoch} with valid_loss: {valid_loss}")
        print(
            f"Epoch {epoch}:  "
            f"Time: {time.time() - epoch_start:.3f}  "
            f"Train Loss: {train_loss:>7.4f}  "
            f"Valid Loss: {valid_loss:>7.4f}  "
            f"Valid Acc: {valid_acc:>7.4f}"
        )
    
if __name__ == "__main__":
    main()
