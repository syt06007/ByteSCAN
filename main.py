import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import EarlyStopper
import dataset
from models.resnets import parse_model_from_name
from models.transformer import AttentionBasedPatchClassifier

# Set seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_folder', type=str, default='ckpt')
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--save', type=bool, default=True, help='Flag to save the model weights')
    parser.add_argument("--model", type=str, default='euclidean_without_avgpool', choices= ["euclidean", "euclidean_without_avgpool", "euclidean_masking"], help="Model name")
    parser.add_argument("--settings", type=str, default='set_num_2', choices=['set_num_1', 'set_num_2', 'set_num_5', 'set_num_6', 'set_num_7'], help="Select settings")
    parser.add_argument("--block_size", type=str, default='4096', choices=['4096', '2048', '1024', '512'], help="Choice dataset size")

    return parser.parse_args()


def load_pretrained_model(checkpoint_path, model, device):
    print('Backbone path : ', checkpoint_path)
    if 'Govdocs' in checkpoint_path.split('/')[-1]:
        checkpoint = torch.load(checkpoint_path, map_location=device)['cnn']
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint)
    return model


def set_loader(args):
    global train
    global valid
    global data_type
    print(f'====================== {args.settings} ======================')
    
    if args.settings == 'set_num_1':
        '''<Train> default setting of FFT75 (4096 data)'''

        # trainer
        from trainer import train as train
        from trainer import valid as valid
        # dataset
        train_data = dataset.SCAN_Dataset(dataset_type='fft', block_size=args.block_size, mode='train', kernel_size=512, overlap=True)
        valid_data = dataset.SCAN_Dataset(dataset_type='fft', block_size=args.block_size, mode='val', kernel_size=512, overlap=True)
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        # models
        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        cnn_backbone = load_pretrained_model('ckpt/backbone/FFT75_RandomCrop_ResNet18.pth', cnn_backbone, args.device)
        model = AttentionBasedPatchClassifier(args, 512, 75).to(args.device)
        data_type='fft'

        return train_loader, valid_loader, cnn_backbone, model

    elif args.settings == 'set_num_2':
        '''<Train> default setting of GovDocs (4096 data)'''

        # trainer
        from trainer import train as train
        from trainer import valid as valid
        # dataset
        train_data = dataset.SCAN_Dataset(dataset_type='govdocs', block_size=args.block_size, mode='train', kernel_size=512, overlap=True)
        valid_data = dataset.SCAN_Dataset(dataset_type='govdocs', block_size=args.block_size, mode='val', kernel_size=512, overlap=True)
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        # models
        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        cnn_backbone = load_pretrained_model('ckpt/backbone/Govdocs_RandomCrop_ResNet18.pth', cnn_backbone, args.device)
        model = AttentionBasedPatchClassifier(args, 512, 75).to(args.device)
        data_type='govdocs'

        return train_loader, valid_loader, cnn_backbone, model

    elif args.settings == 'set_num_5':
        '''<Train> ablation w/o random crop (Fixed Crop) (1024, 2048, 4096 data)'''

        # trainer
        from trainer import train as train
        from trainer import valid as valid
        # dataset
        train_data = dataset.SCAN_Dataset(dataset_type='fft', block_size=args.block_size, mode='train', kernel_size=512, overlap=True)
        valid_data = dataset.SCAN_Dataset(dataset_type='fft', block_size=args.block_size, mode='val', kernel_size=512, overlap=True)
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        # models
        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        cnn_backbone = load_pretrained_model('ckpt/backbone/FFT75_FixedCrop_ResNet18.pth', cnn_backbone, args.device)
        model = AttentionBasedPatchClassifier(args, 512, 75).to(args.device)
        data_type='fft'

        return train_loader, valid_loader, cnn_backbone, model

    elif args.settings == 'set_num_6':
        '''<Train> ablation w/o overlapping patch (kernel_size == 512 byte) (1024, 2048, 4096 data)'''

        # trainer
        from trainer import train as train
        from trainer import valid as valid
        # dataset
        train_data = dataset.SCAN_Dataset(dataset_type='fft', block_size=args.block_size, mode='train', kernel_size=512, overlap=False)
        valid_data = dataset.SCAN_Dataset(dataset_type='fft', block_size=args.block_size, mode='val', kernel_size=512, overlap=False)
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        # models
        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        cnn_backbone = load_pretrained_model('ckpt/backbone/FFT75_RandomCrop_ResNet18.pth', cnn_backbone, args.device)
        model = AttentionBasedPatchClassifier(args, 512, 75).to(args.device)
        data_type='fft'

        return train_loader, valid_loader, cnn_backbone, model

    elif args.settings == 'set_num_7':
        '''<Train> ablation w/o overlapping patch (kernel_size == 256 byte) (1024, 2048, 4096 data)'''

        # trainer
        from trainer import train as train
        from trainer import valid as valid
        # dataset
        train_data = dataset.SCAN_Dataset(dataset_type='fft', block_size=args.block_size, mode='train', kernel_size=256, overlap=False)
        valid_data = dataset.SCAN_Dataset(dataset_type='fft', block_size=args.block_size, mode='val', kernel_size=256, overlap=False)
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        # models
        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        cnn_backbone = load_pretrained_model('ckpt/backbone/FFT75_RandomCrop_ResNet18.pth', cnn_backbone, args.device)
        model = AttentionBasedPatchClassifier(args, 512, 75).to(args.device)
        data_type='fft'

        return train_loader, valid_loader, cnn_backbone, model


def main():
    args = parse_args()

    # Create some strings for file management
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(dir_path, args.save_folder)
    print(f"Experiment directory: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)

    train_loader, valid_loader, cnn_backbone, model = set_loader(args)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
    early_stopper = EarlyStopper(patience=11, verbose=True)
    criterion = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train for one epoch
        train_loss = train(args, train_loader, cnn_backbone, model, criterion, optimizer)
        # Validate for one epoch
        valid_loss, valid_acc = valid(args, valid_loader, cnn_backbone, model, scheduler, early_stopper)
        
        print(f'valid_loss : {valid_loss}, valid_acc : {valid_acc}')
        
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
                'transformer': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch}
            if args.save:
                save_path = os.path.join(exp_dir, f"{data_type}/{args.settings}_epoch_{epoch}_BestLoss.pth")
                torch.save(best_model_state, save_path)
                print(f"Model saved at epoch {epoch} with valid_loss: {valid_loss}")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model_state = {
                'cnn': cnn_backbone.state_dict(),
                'transformer': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch}
            if args.save:
                save_path = os.path.join(exp_dir, f"{data_type}/{args.settings}_epoch_{epoch}_BestAcc.pth")
                torch.save(best_model_state, save_path)
                print(f"Model saved at epoch {epoch} with valid_acc: {valid_acc}")

        print(
            f"Epoch {epoch}:  "
            f"Time: {time.time() - epoch_start:.3f}  "
            f"Train Loss: {train_loss:>7.4f}  "
            f"Valid Loss: {valid_loss:>7.4f}  "
            f"Valid Acc: {valid_acc:>7.4f}"
        )

    # Store metrics
    metrics = {"valid_loss": valid_loss, "valid_acc": valid_acc}
    with open(os.path.join(exp_dir, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=4)


if __name__ == "__main__":
    main()
