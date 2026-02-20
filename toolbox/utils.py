import torch
import torch.nn.functional as F

def evaluate_model(model, loader):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            loss = F.cross_entropy(outputs[-1], targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs[-1].data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float().item()
            b_idx = batch_idx
    print(f'TEST | Loss: {val_loss/(b_idx+1):.3f} | Acc: {100*correct/total:.3f} |')
    return val_loss/(b_idx+1), correct*100/total