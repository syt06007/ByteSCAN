import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import EarlyStopper
# from trainer import train, valid
from dataset import Fifty_dataset  # 새로운 데이터셋 클래스
import dataset

from models.resnets import parse_model_from_name

# Set seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--data_dir', type=str, default='../../Data/FFT75')
    parser.add_argument('--save_folder', type=str, default='resnet_ckpt')
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--opt', type=str, default='adam', help='Optimizer type')
    parser.add_argument('--save', type=bool, default=True, help='Flag to save the model weights')
    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint name for saving models')
    parser.add_argument('--classes', type=str, default=None, help='Comma-separated list of class indices to include')

    parser.add_argument('--block_size', type=str, default='4k', help='type : str >>  512 or 4k')
    parser.add_argument("--model", type=str, default='euclidean_without_avgpool', choices= ["euclidean", "euclidean_without_avgpool", "euclidean_masking"], help="Model name")
    parser.add_argument("--dataset", type=str, default='', choices=["Normal, 4KCropEnsembleSliding, 4KCropEnsemble256, 4KCropEnsemble"], help="Dataset (fifty)")
    parser.add_argument('--num_heads', type=int, default=8, help='Number of Transformer heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of Transformer layers')

    parser.add_argument('--memo', type=str, default='RN18_4KCropped512_baseline_training', help='memo')
    return parser.parse_args()



def set_loader(args, mode='train'):
    global train
    global valid

    Fifty_dataset
    data = dataset.Fifty_dataset(args, mode=mode)
    # data = dataset.Fifty4KCrop_v2(args, mode=mode)
    # data = dataset.Fifty4KCrop_RandomCropMasking(args, mode=mode)
    # data = dataset.GovDatasetRandomCrop(args, mode=mode)
    from trainer import gov_train as train
    from trainer import gov_valid as valid


    if mode=='train':
        data_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=args.batch_size)
    else:
        data_loader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=args.batch_size)
    return data_loader



def main():
    args = parse_args()

    # Scenario에 따른 클래스 설정
    if args.classes is None:
        class_lst = [75, 11, 25, 5, 2, 2]
        classes = class_lst[args.scenario - 1]
    else:
        classes = len(args.classes.split(','))
        
    print(f"Num Classes: {classes}")

    # Initialize wandb
    project_name = f'{args.dataset}_{args.memo}'    
    args.checkpoint = args.checkpoint + project_name


    print(f"==========Scenario {args.scenario} ==========")
    print(f"=========={args.model} ==========")
    print(f"=========={args.checkpoint} ==========")


    # Create some strings for file management
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(dir_path, args.save_folder)
    print(f"Experiment directory: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)

    train_loader = set_loader(args, mode='train')
    valid_loader = set_loader(args, mode='val')

    # Create model
    cnn_backbones = parse_model_from_name(args.model, 75).to(args.device)
    # cnn_backbones = load_pretrained_model("resnet_ckpt/ResNet_base_KD_4KCropRandom_4k_crop_512_training_baseline.pth", cnn_backbones, args.device)
    
    # model = CNNTransformerModel(args).to(args.device) 
    # model = AttentionBasedPatchClassifier(args, 512, 75).to(args.device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(cnn_backbones.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
    early_stopper = EarlyStopper(patience=11, verbose=True)

    print(f"Using optimizer: {optimizer}")

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(gamma=2)  # Use FocalLoss instead of CrossEntropyLoss

    best_valid_loss = float('inf')
    if args.wandb:
            wandb.init(project="Resnet_Euc", entity='syt06007', config=args)
            wandb.run.name = project_name
            config = wandb.config
            # Print initial messages
    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train for one epoch
        train_loss = train(args, train_loader, cnn_backbones, criterion, optimizer, epoch)

        # Validate for one epoch
        valid_loss, valid_acc = valid(args, valid_loader, cnn_backbones, scheduler, early_stopper)
        print(f'valid_loss : {valid_loss}, valid_acc : {valid_acc}')
        # Log metrics to wandb
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        # Early stopping check
        print(f"Calling early_stopper for epoch {epoch}")  # 디버깅 출력문 추가
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Save the best model if valid_loss decreases
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = {
                'cnn': cnn_backbones.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            if args.save:
                save_path = os.path.join(exp_dir, f"{args.checkpoint}.pth")
                torch.save(best_model_state, save_path)
                print(f"Model saved at epoch {epoch} with valid_loss: {valid_loss}")
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

    # Finish the wandb run
    if args.wandb:
        wandb.finish()


def load_pretrained_model(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    checkpoint_dict = {}
    
    for k, v in checkpoint.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                checkpoint_dict[k] = v
            elif model_dict[k].shape == v.unsqueeze(0).shape:  # 예: (128,) -> (1, 128)
                checkpoint_dict[k] = v.unsqueeze(0)
                print(f"{k}의 형태를 {v.shape}에서 {v.unsqueeze(0).shape}로 변환")
            elif model_dict[k].shape == v.squeeze(0).shape:  # 예: (1, 128) -> (128,)
                checkpoint_dict[k] = v.squeeze(0)
                print(f"{k}의 형태를 {v.shape}에서 {v.squeeze(0).shape}로 변환")
            else:
                print(f"{k}의 크기 불일치: 체크포인트 형태 {v.shape}, 모델 형태 {model_dict[k].shape}")
    
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    return model



if __name__ == "__main__":
    main()
