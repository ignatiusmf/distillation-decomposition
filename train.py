from toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby
from toolbox.data_loader import Cifar10, Cifar100, TinyImageNet
from toolbox.utils import plot_the_things, evaluate_model

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random

from pathlib import Path
import argparse

DEVICE = "cuda"
EPOCHS = 150
BATCH_SIZE = 128

MODELS = {
    'ResNet112': ResNet112,
    'ResNet56': ResNet56,
    'ResNet20': ResNet20,
    'ResNetBaby': ResNetBaby,
}

DATASETS = {
    'TinyImageNet': TinyImageNet,
    'Cifar100': Cifar100,
    'Cifar10': Cifar10
}

parser = argparse.ArgumentParser(description='Run a training script with custom parameters.')
parser.add_argument('--experiment_name', type=str, default='default')
parser.add_argument('--model', type=str, default='ResNet112', choices=MODELS.keys())
parser.add_argument('--dataset', type=str, default='Cifar100', choices=DATASETS.keys())
args = parser.parse_args()

EXPERIMENT_PATH = args.experiment_name
MODEL = args.model
DATASET = args.dataset
seed = int(EXPERIMENT_PATH.split('/')[-1]) if EXPERIMENT_PATH.split('/')[-1].isdigit() else 0

Path(f"experiments/{EXPERIMENT_PATH}").mkdir(parents=True, exist_ok=True)

print(vars(args), f'{seed=}')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Data = DATASETS[DATASET](BATCH_SIZE, seed=seed)
trainloader, testloader = Data.trainloader, Data.testloader

model = MODELS[MODEL](Data.class_num).to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_loss = []
train_acc = []
test_loss = []
test_acc = []
max_acc = 0.0

for i in range(EPOCHS):
    print("Epoch", i)
    model.train()
    val_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs[3], targets, label_smoothing=0.1)
        loss.backward()
        optimizer.step()
        val_loss += loss.item()
        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    scheduler.step()

    print(f'TRAIN | Loss: {val_loss/(b_idx+1):.3f} | Acc: {100*correct/total:.2f} |')
    tel, tea = evaluate_model(model, testloader)
    train_loss.append(val_loss/(b_idx+1))
    train_acc.append(100*correct/total)
    test_loss.append(tel)
    test_acc.append(tea)

    if tea > max_acc:
        max_acc = tea
        checkpoint_name = '_'.join(EXPERIMENT_PATH.split('/')[:-1])
        torch.save({'weights': model.state_dict()}, f'experiments/{EXPERIMENT_PATH}/{checkpoint_name}.pth')
    
    plot_the_things(train_loss, test_loss, train_acc, test_acc, EXPERIMENT_PATH)

import json

with open(f'experiments/{EXPERIMENT_PATH}/metrics.json', 'w') as f:
    json.dump({
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    }, f)