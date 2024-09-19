import torch
import torch.nn as nn
from utils import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from einops import rearrange


def train(cfg, train_loader, feature_extractor, transformer, criterion, optimizer):
    """ One epoch training """
    feature_extractor.eval()
    transformer.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    pbar = tqdm(train_loader)
    for idx, (data, label) in enumerate(pbar):
        data, label = data.type(torch.LongTensor).to(cfg.device), label.long().to(cfg.device)
        
        # Forward
        inputs = rearrange(data, 'b n p -> (b n) p') # batch num_views, patchsize
        with torch.no_grad():
            _, embedding = feature_extractor(inputs)
        
        all_embeddings = rearrange(embedding, '(b n) c l -> b (n l) c', b=data.shape[0], n=int(embedding.shape[0]/data.shape[0]))
        logits = transformer(all_embeddings)

        loss = criterion(logits, label)
        
        _, pred_label = torch.max(logits, 1)
        acc = accuracy_score(label.cpu().numpy(), pred_label.cpu().numpy())

        # Update metric
        losses.update(loss.item(), label.size(0))
        accuracies.update(acc, label.size(0))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=losses.avg, acc=accuracies.avg)

    return losses.avg



def valid(cfg, valid_loader, feature_extractor, transformer, scheduler, early_stopper):
    feature_extractor.eval()
    transformer.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(valid_loader)
    with torch.no_grad():
        for idx, (data, label) in enumerate(pbar):
            data, label = data.type(torch.LongTensor).to(cfg.device), label.long().to(cfg.device)

            if len(data.shape) == 2: # if 512 dataset (b, seq)
                data = data.unsqueeze(1) # (b, 1, seq)
            inputs = rearrange(data, 'b n p -> (b n) p') # batch num_views, patchsize
            with torch.no_grad():
                _, embedding = feature_extractor(inputs)
                all_embeddings = rearrange(embedding, '(b n) c l -> b (n l) c', b=data.shape[0], n=int(embedding.shape[0]/data.shape[0]))
                logits = transformer(all_embeddings)

            loss = criterion(logits, label)
            # Compute acc
            _, pred_label = torch.max(logits, 1)
            acc = accuracy_score(label.cpu().numpy(), pred_label.cpu().numpy())

            # Update metric
            losses.update(loss.item(), label.size(0))
            accuracies.update(acc, label.size(0))
            pbar.set_postfix(loss=losses.avg, acc=accuracies.avg)

    # callbacks
    scheduler.step(losses.avg)
    early_stopper(losses.avg)

    return losses.avg, accuracies.avg

def test(cfg, test_loader, feature_extractor, transformer):
    feature_extractor.eval()
    transformer.eval()

    accuracies = AverageMeter()


    pbar = tqdm(test_loader)
    with torch.no_grad():
        for idx, (data, label) in enumerate(pbar):
            data, label = data.type(torch.LongTensor).to(cfg.device), label.long().to(cfg.device)

            if len(data.shape) == 2: # if 512 dataset (b, seq)
                data = data.unsqueeze(1) # (b, 1, seq)
            inputs = rearrange(data, 'b n p -> (b n) p') # batch num_views, patchsize
            with torch.no_grad():
                _, embedding = feature_extractor(inputs)
                all_embeddings = rearrange(embedding, '(b n) c l -> b (n l) c', b=data.shape[0], n=int(embedding.shape[0]/data.shape[0]))
                logits = transformer(all_embeddings)

            # Compute acc
            _, pred_label = torch.max(logits, 1)
            acc = accuracy_score(label.cpu().numpy(), pred_label.cpu().numpy())

            # Update metric
            accuracies.update(acc, label.size(0))
            pbar.set_postfix(acc=accuracies.avg)

    print(f'Acc : {acc.avg}')


def soft_voting(cfg, test_loader, feature_extractor, transformer):
    feature_extractor.eval()
    transformer.eval()

    accuracies = AverageMeter()


    pbar = tqdm(test_loader)
    with torch.no_grad():
        for idx, (data, label) in enumerate(pbar):
            data, label = data.type(torch.LongTensor).to(cfg.device), label.long().to(cfg.device)

            if len(data.shape) == 2: # if 512 dataset (b, seq)
                data = data.unsqueeze(1) # (b, 1, seq)
            inputs = rearrange(data, 'b n p -> (b n) p') # batch num_views, patchsize
            with torch.no_grad():
                logit, _ = feature_extractor(inputs)
                logits = rearrange(logit, '(b n) c -> b n c', b=data.shape[0], n=int(logit.shape[0]/data.shape[0]))

            logits = logits.mean(dim=1)
            # Compute acc
            _, pred_label = torch.max(logits, 1)
            acc = accuracy_score(label.cpu().numpy(), pred_label.cpu().numpy())

            # Update metric
            accuracies.update(acc, label.size(0))
            pbar.set_postfix(acc=accuracies.avg)

    print(f'Acc : {acc.avg}')


def test_var_size(cfg, valid_loader, feature_extractor, transformer):
    feature_extractor.eval()
    transformer.eval()

    accuracies512 = AverageMeter()
    accuracies1024 = AverageMeter()
    accuracies2048 = AverageMeter()

    pbar = tqdm(valid_loader)
    acc_list = [accuracies512, accuracies1024, accuracies2048]

    with torch.no_grad():
        for idx, (data_512, data_1024, data_2048, label) in enumerate(pbar):
            data_512, data_1024, data_2048, label = data_512.type(torch.LongTensor).to(cfg.device), data_1024.type(torch.LongTensor).to(cfg.device), \
                                                            data_2048.type(torch.LongTensor).to(cfg.device), label.long().to(cfg.device)

            inputs512 = rearrange(data_512, 'b n p -> (b n) p') # batch num_views, patchsize
            inputs1024 = rearrange(data_1024, 'b n p -> (b n) p') # batch num_views, patchsize
            inputs2048 = rearrange(data_2048, 'b n p -> (b n) p') # batch num_views, patchsize
            input_list = [inputs512, inputs1024, inputs2048]

            for i, data in enumerate(input_list):
                _, embedding = feature_extractor(data)

                all_embeddings = rearrange(embedding, '(b n) c l -> b (n l) c', b=data.shape[0], n=int(embedding.shape[0]/data.shape[0]))
                logits = transformer(all_embeddings)

                # Compute acc
                _, pred_label = torch.max(logits, 1)
                acc = accuracy_score(label.cpu().numpy(), pred_label.cpu().numpy())
                acc_list[i].update(acc, label.size(0))

    print('====== dataset accuracies ======')
    print(f'512 : {acc_list[0].avg}, 1024 : {acc_list[1].avg}, 2048 : {acc_list[2].avg}')

