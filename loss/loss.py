import torch.nn as nn
import torch

def calc_cross_entroply_loss(input, target, model, device):
    input = input.to(device) # shape of (B, S, E)
    target = target.to(device) # shape of (B, S)

    logits = model(input) # shape of (B, S, vocab_size)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), 
                                       target.flatten())
    return loss

def calc_loss_batch(input, target, model, device):
    input = input.to(device) 
    target = target.to(device)
    logits = model(input)[:,-1,:]
    loss = nn.functional.cross_entropy(logits, 
                                       target)
    return loss

def calc_loss_loader_ch5(dataloader, model, device, num_batches=None):
    total_loss = 0
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (input, target) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_cross_entroply_loss(input, target, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def calc_accuracy_loader(dataloader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1 ,:] # only using the last token which is richest in info
            pred_label = torch.argmax(logits, dim=-1)
            num_examples += pred_label.shape[0]
            correct_predictions += ((pred_label == target_batch).sum().item())
        else:
            break
    return correct_predictions / num_batches

def calc_loss_loader(dataloader, model, device, num_batches=None):
    total_loss = 0
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (input, target) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input, target, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches