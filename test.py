import argparse
import random

import numpy as np
import torch

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
    parser.add_argument("--settings", type=str, default='set_num_4', choices=['set_num_1', 'set_num_2', 'set_num_3', 'set_num_4', 'set_num_5', 'set_num_6', 'set_num_7'], help="Dataset (fifty)")
    parser.add_argument("--block_size", type=str, default='4096', choices=['4096', '2048', '1024', '512'], help="Choice dataset size")
    return parser.parse_args()


def load_pretrained_model(checkpoint_path, model, device):
    if 'Govdocs' in checkpoint_path.split('/')[-1]:
        checkpoint = torch.load(checkpoint_path, map_location=device)['cnn']
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint)
    return model


def set_loader(args):
    global valid
    global data_type

    if args.settings == 'set_num_1':
        '''<Test> Test setting of FFT75'''

        # trainer
        from trainer import test as valid
        # dataset
        valid_data = dataset.SCAN_Dataset(dataset_type='fft', block_size='4096', mode='val', kernel_size=512, overlap=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        # models
        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        cnn_backbone = load_pretrained_model('ckpt/backbone/FFT75_RandomCrop_ResNet18.pth', cnn_backbone, args.device)
        model = AttentionBasedPatchClassifier(args, 512, 75).to(args.device)
        model_ckpt = torch.load(checkpoint_path, map_location=device)
        data_type='fft'

        return valid_loader, cnn_backbone, model

    elif args.settings == 'set_num_2':
        '''<Test> Test setting of GovDocs'''

        # trainer
        from trainer import test as valid
        # dataset
        valid_data = dataset.SCAN_Dataset(dataset_type='govdocs', block_size='4096', mode='val', kernel_size=512, overlap=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        # models
        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        cnn_backbone = load_pretrained_model('ckpt/backbone/Govdocs_RandomCrop_ResNet18.pth', cnn_backbone, args.device)
        model = AttentionBasedPatchClassifier(args, 512, 75).to(args.device)
        model_ckpt = torch.load(checkpoint_path, map_location=device)
        data_type='govdocs'

        return valid_loader, cnn_backbone, model


    elif args.settings == 'set_num_3':
        '''<Test> data size variation (512, 1024, 2048 data)'''

        # trainer
        from trainer import test_var_size as valid
        # dataset
        valid_data = dataset.SCAN_Dataset_VarSize(dataset_type='govdocs', block_size='4096', mode='val', kernel_size=512, overlap=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        # models
        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        cnn_backbone = load_pretrained_model('ckpt/backbone/FFT75_RandomCrop_ResNet18.pth', cnn_backbone, args.device)
        model = AttentionBasedPatchClassifier(args, 512, 75).to(args.device)
        model_ckpt = torch.load(checkpoint_path, map_location=device)
        data_type='fft'

        return valid_loader, cnn_backbone, model


    elif args.settings == 'set_num_4':
        '''<Test> ablation w/o attention (soft voting) (1024, 2048, 4096 data)'''

        # trainer
        from trainer import soft_voting as valid
        # dataset
        valid_data = dataset.SCAN_Dataset(dataset_type='fft', block_size='4096', mode='val', kernel_size=512, overlap=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        # models
        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        cnn_backbone = load_pretrained_model('ckpt/backbone/FFT75_RandomCrop_ResNet18.pth', cnn_backbone, args.device)
        model = None
        data_type='fft'

        return valid_loader, cnn_backbone, model

    elif args.settings == 'set_num_5':
        '''<Test> ablation w/o random crop (Fixed Crop) (1024, 2048, 4096 data)'''

        # trainer
        from trainer import test as valid
        # dataset
        valid_data = dataset.SCAN_Dataset(dataset_type='fft', block_size='4096', mode='val', kernel_size=512, overlap=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        # models
        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        cnn_backbone = load_pretrained_model('ckpt/backbone/FFT75_FixedCrop_ResNet18.pth', cnn_backbone, args.device)
        model = AttentionBasedPatchClassifier(args, 512, 75).to(args.device)
        model_ckpt = torch.load(checkpoint_path, map_location=device)
        data_type='fft'

        return valid_loader, cnn_backbone, model

    elif args.settings == 'set_num_6':
        '''<Test> ablation w/o overlapping patch (kernel_size==512 or 256 byte) (1024, 2048, 4096 data)'''

        # trainer
        from trainer import test as valid
        # dataset
        valid_data = dataset.SCAN_Dataset(dataset_type='fft', block_size='4096', mode='val', kernel_size=512, overlap=False)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        # models
        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        cnn_backbone = load_pretrained_model('ckpt/backbone/.pth', cnn_backbone, args.device)
        model = AttentionBasedPatchClassifier(args, 512, 75).to(args.device)
        model_ckpt = torch.load(checkpoint_path, map_location=device)
        data_type='fft'

        return valid_loader, cnn_backbone, model

    elif args.settings == 'set_num_7':
        '''<Test> ablation w/o overlapping patch (kernel_size==512 or 256 byte) (1024, 2048, 4096 data)'''

        # trainer
        from trainer import test as valid
        # dataset
        valid_data = dataset.SCAN_Dataset(dataset_type='fft', block_size='4096', mode='val', kernel_size=256, overlap=False)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        # models
        cnn_backbone = parse_model_from_name(args.model, 75).to(args.device)
        cnn_backbone = load_pretrained_model('ckpt/backbone/.pth', cnn_backbone, args.device)
        model = AttentionBasedPatchClassifier(args, 512, 75).to(args.device)
        model_ckpt = torch.load(checkpoint_path, map_location=device)
        data_type='fft'

        return valid_loader, cnn_backbone, model


def main():
    args = parse_args()
    valid_loader, cnn_backbone, model = set_loader(args)

    valid(args, valid_loader, cnn_backbone, model)

if __name__ == "__main__":
    main()
