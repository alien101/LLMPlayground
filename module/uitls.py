import torch
from loss.loss import *
from module.GPT.model import *
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        valid_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()

    return train_loss, valid_loss

def generate_and_print_sample(model, tokenizer, device, start_context, temperature=None, k=None):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token(start_context, tokenizer).to(device)

    with torch.no_grad():
        if temperature is None:
            token_ids = greedy_inference(model=model,
                                idx=encoded,
                                max_new_tokens=50,
                                context_size=context_size)
        else:
            token_ids = generate(model=model,
                                idx=encoded,
                                max_new_tokens=50,
                                context_size=context_size,
                                temperature=temperature,
                                topk=k
                                )           
    decoded = token_to_text(token_ids, tokenizer)
    print(decoded.replace("\n", ""))
    model.train()

