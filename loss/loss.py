import torch.nn as nn

def calc_cross_entroply_loss(input, target, model, device):
    input = input.to(device) # shape of (B, S, E)
    target = target.to(device) # shape of (B, S)

    logits = model(input) # shape of (B, S, vocab_size)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), 
                                       target.flatten())
    return loss

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
            loss = calc_cross_entroply_loss(input, target, model, device)
            total_loss += loss
        else:
            break
    return total_loss / num_batches