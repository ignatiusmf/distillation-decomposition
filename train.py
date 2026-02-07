from toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby
from toolbox.data_loader import Cifar10, Cifar100, TinyImageNet
from toolbox.utils import plot_the_things, evaluate_model
from toolbox.distillation import get_distillation_method, DISTILLATION_METHODS

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random

from pathlib import Path
import argparse
import json
import sys

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
parser.add_argument('--model', type=str, default='ResNet112', choices=MODELS.keys())
parser.add_argument('--dataset', type=str, default='Cifar100', choices=DATASETS.keys())
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

# Distillation arguments
parser.add_argument('--distillation', type=str, default='none', choices=DISTILLATION_METHODS.keys(),
                    help='Distillation method to use')
parser.add_argument('--teacher_model', type=str, default=None, choices=list(MODELS.keys()) + [None],
                    help='Teacher model architecture (required if distillation is not none)')
parser.add_argument('--teacher_weights', type=str, default=None,
                    help='Path to teacher model weights')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight for distillation loss (final_loss = (1-alpha)*CE + alpha*distill_loss)')
parser.add_argument('--temperature', type=float, default=4.0,
                    help='Temperature for logit distillation')

# Resume/checkpoint arguments
parser.add_argument('--force_restart', action='store_true',
                    help='Force restart training even if checkpoint exists')

args = parser.parse_args()

# Validate distillation arguments
if args.distillation != 'none':
    if args.teacher_model is None or args.teacher_weights is None:
        parser.error("--teacher_model and --teacher_weights required when using distillation")

MODEL = args.model
DATASET = args.dataset
seed = args.seed

# Setup directories - everything goes in experiments/
# Structure: experiments/[dataset]/[method]/[model_name]/[seed]/
if args.distillation == 'none':
    experiment_dir = Path(f'experiments/{DATASET}/pure/{MODEL}/{seed}')
else:
    experiment_dir = Path(f'experiments/{DATASET}/{args.distillation}/{args.teacher_model}_to_{MODEL}/{seed}')
experiment_dir.mkdir(parents=True, exist_ok=True)

# Checkpoint paths (all in experiment_dir)
checkpoint_path = experiment_dir / 'checkpoint.pth'  # Latest training state
best_model_path = experiment_dir / 'best.pth'        # Best model weights only
status_path = experiment_dir / 'status.json'         # Training status

def load_status():
    """Load training status from file."""
    if status_path.exists():
        with open(status_path, 'r') as f:
            return json.load(f)
    return {'status': 'not_started', 'epoch': 0, 'max_acc': 0.0}

def save_status(status_dict):
    """Save training status to file."""
    with open(status_path, 'w') as f:
        json.dump(status_dict, f, indent=2)

# Check if training is already complete
status = load_status()
if status['status'] == 'completed' and not args.force_restart:
    print(f"Training already completed for {experiment_dir}")
    print(f"  Best accuracy: {status['max_acc']:.2f}%")
    print(f"  Use --force_restart to retrain from scratch")
    sys.exit(0)

print(vars(args), f'{seed=}')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Data = DATASETS[DATASET](BATCH_SIZE, seed=seed)
trainloader, testloader = Data.trainloader, Data.testloader

# Create student model
model = MODELS[MODEL](Data.class_num).to(DEVICE)

# Load teacher model if doing distillation
teacher_model = None
if args.distillation != 'none':
    teacher_model = MODELS[args.teacher_model](Data.class_num).to(DEVICE)
    checkpoint = torch.load(args.teacher_weights, map_location=DEVICE, weights_only=True)
    teacher_model.load_state_dict(checkpoint['weights'])
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    print(f"Loaded teacher model: {args.teacher_model} from {args.teacher_weights}")

# Create distillation method
distillation = get_distillation_method(
    args.distillation,
    student_model=model,
    teacher_model=teacher_model,
    alpha=args.alpha,
    temperature=args.temperature
).to(DEVICE)

# Collect all trainable parameters (student + any distillation modules)
trainable_params = list(model.parameters())
for module in distillation.get_trainable_modules():
    trainable_params.extend(module.parameters())

optimizer = optim.SGD(trainable_params, lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Initialize training state
train_loss = []
train_acc = []
test_loss = []
test_acc = []
max_acc = 0.0
start_epoch = 0

# Try to resume from checkpoint
if checkpoint_path.exists() and not args.force_restart:
    print(f"Found checkpoint at {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    
    # Load distillation module states if present
    if 'distillation_state_dict' in ckpt and ckpt['distillation_state_dict'] is not None:
        for i, module in enumerate(distillation.get_trainable_modules()):
            module.load_state_dict(ckpt['distillation_state_dict'][i])
    
    start_epoch = ckpt['epoch'] + 1
    train_loss = ckpt['train_loss']
    train_acc = ckpt['train_acc']
    test_loss = ckpt['test_loss']
    test_acc = ckpt['test_acc']
    max_acc = ckpt['max_acc']
    
    print(f"Resuming from epoch {start_epoch}, max_acc={max_acc:.2f}%")
elif args.force_restart and checkpoint_path.exists():
    print("Force restart: ignoring existing checkpoint")

# Update status to in_progress
save_status({'status': 'in_progress', 'epoch': start_epoch, 'max_acc': max_acc, 'config': vars(args)})

for i in range(start_epoch, EPOCHS):
    print("Epoch", i)
    model.train()
    val_loss, correct, total = 0, 0, 0
    b_idx = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        
        # Forward pass through student
        outputs = model(inputs)
        
        # Cross-entropy loss with label smoothing
        ce_loss = F.cross_entropy(outputs[-1], targets, label_smoothing=0.1)
        
        # Distillation loss
        if args.distillation != 'none':
            assert teacher_model is not None
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            distill_loss = distillation.extra_loss(outputs, teacher_outputs, targets)
            loss = (1 - distillation.alpha) * ce_loss + distillation.alpha * distill_loss
        else:
            loss = ce_loss
        
        loss.backward()
        optimizer.step()
        val_loss += loss.item()
        _, predicted = torch.max(outputs[-1].data, 1)
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

    # Save best model (weights only, for inference/distillation)
    if tea > max_acc:
        max_acc = tea
        torch.save({'weights': model.state_dict()}, best_model_path)
        print(f"Saved best model to {best_model_path}")
    
    # Save checkpoint (full training state for resumption)
    distillation_state = None
    if distillation.get_trainable_modules():
        distillation_state = [m.state_dict() for m in distillation.get_trainable_modules()]
    
    torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'distillation_state_dict': distillation_state,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'max_acc': max_acc,
        'config': vars(args)
    }, checkpoint_path)
    
    # Update status
    save_status({'status': 'in_progress', 'epoch': i, 'max_acc': max_acc, 'config': vars(args)})
    
    plot_the_things(train_loss, test_loss, train_acc, test_acc, experiment_dir)

# Mark training as completed
save_status({'status': 'completed', 'epoch': EPOCHS, 'max_acc': max_acc, 'config': vars(args)})
print(f"\nTraining completed! Best accuracy: {max_acc:.2f}%")
print(f"Best model saved to: {best_model_path}")

with open(experiment_dir / 'metrics.json', 'w') as f:
    json.dump({
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'config': vars(args)
    }, f)