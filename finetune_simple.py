import torch
import numpy as np
from loss.loss import *
from module.GPT.gpt_download import download_and_load_gpt2
from torch.utils.data import DataLoader
import tiktoken
from datamodule.dataset import *
from module.uitls import *
from architecture.config import *

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: Left {left.shape}, Right {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)

        gpt.transform_blocks[b].attn.W_k.weight = assign(gpt.transform_blocks[b].attn.W_k.weight, k_w.T)
        gpt.transform_blocks[b].attn.W_q.weight = assign(gpt.transform_blocks[b].attn.W_q.weight, q_w.T)
        gpt.transform_blocks[b].attn.W_v.weight = assign(gpt.transform_blocks[b].attn.W_v.weight, v_w.T)

        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)

        gpt.transform_blocks[b].attn.W_k.bias = assign(gpt.transform_blocks[b].attn.W_k.bias, k_b)
        gpt.transform_blocks[b].attn.W_q.bias = assign(gpt.transform_blocks[b].attn.W_q.bias, q_b)
        gpt.transform_blocks[b].attn.W_v.bias = assign(gpt.transform_blocks[b].attn.W_v.bias, v_b)

        gpt.transform_blocks[b].attn.combine_head.weight = assign(gpt.transform_blocks[b].attn.combine_head.weight,
                                                                  params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transform_blocks[b].attn.combine_head.bias = assign(gpt.transform_blocks[b].attn.combine_head.bias,
                                                                  params["blocks"][b]["attn"]["c_proj"]["b"])
        
        gpt.transform_blocks[b].ff.layers[0].weight = assign(gpt.transform_blocks[b].ff.layers[0].weight,
                                                             params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transform_blocks[b].ff.layers[0].bias = assign(gpt.transform_blocks[b].ff.layers[0].bias,
                                                           params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transform_blocks[b].ff.layers[2].weight = assign(gpt.transform_blocks[b].ff.layers[2].weight,
                                                             params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transform_blocks[b].ff.layers[2].bias = assign(gpt.transform_blocks[b].ff.layers[2].bias,
                                                           params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transform_blocks[b].norm1.scale = assign(gpt.transform_blocks[b].norm1.scale,
                                                    params["blocks"][b]["ln_1"]["g"])
        gpt.transform_blocks[b].norm1.shift = assign(gpt.transform_blocks[b].norm1.shift,
                                                    params["blocks"][b]["ln_1"]["b"])
        gpt.transform_blocks[b].norm2.scale = assign(gpt.transform_blocks[b].norm2.scale,
                                                    params["blocks"][b]["ln_2"]["g"])
        gpt.transform_blocks[b].norm2.shift = assign(gpt.transform_blocks[b].norm2.shift,
                                                    params["blocks"][b]["ln_2"]["b"])     

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])  
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out.weight = assign(gpt.out.weight, params["wte"])

def train_classifier_simple(model, 
                            train_loader, val_loader, 
                            optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [],[],[],[]
    example_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input, label in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input, label, model, device)
            loss.backward()
            optimizer.step()
            example_seen += input.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(model,train_loader, val_loader,
                                                        device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(valid_loss)
                print(f"Epoch{epoch + 1} (step {global_step:06d}): "
                      f"Train loss {train_loss:.3f} "
                      f"Val loss {valid_loss:.3f}"
                )
        train_acc= calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_acc= calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"train acc: {train_acc}, valid acc: {val_acc}")

        train_accs.append(train_acc)
        val_accs.append(val_acc)
    return train_losses, val_losses, train_accs, val_accs, example_seen

def classify_review(text, model, tokenizer, 
                    device, max_length=None, pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_eonctext_length = model.pos_emb.weight.shape[1]

    input_ids = input_ids[:min(max_length, supported_eonctext_length)]
    input_ids += [pad_token_id] * (max_length - supported_eonctext_length)
    
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"
    pass

if __name__ == "__main__":
    # Fine tune spam/ not spam from gpt
    torch.manual_seed(123)
    tokenizer=tiktoken.get_encoding("gpt2")
    csvs = [f"csv/{status}.csv" for status in ["train","valid","test"]]
    datasets=[]
    loaders=[]
    num_workers = 0
    batch_size = 10

    for csv in csvs:
        datasets.append(SpamDataset(
            csv,
            max_length=None,
            tokenizer=tokenizer
        ))

    for dataset in datasets:
        loaders.append(DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False
        ))
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    gpt = GPT(GPT_CONFIG_124M)
    load_weights_into_gpt(gpt, params)

    start_context="Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    for param in gpt.parameters():
        param.requires_grad = False
    gpt.out=torch.nn.Linear(
        in_features=GPT_CONFIG_124M["emb_dim"],
        out_features=2
    )
    for param in gpt.transform_blocks[-1].parameters():
        param.requires_grad = True
    for param in gpt.final_norm.parameters():
        param.requires_grad = True
    device = torch.device("cuda")
    gpt.cuda()

    optimizer = torch.optim.AdamW(gpt.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs=5
    res = train_classifier_simple(gpt, loaders[0], loaders[1],
                            optimizer, device, num_epochs,
                            eval_freq=50, eval_iter=5)
    
    text1 = ("You are a winner you have been specially selected to receive $ 1000 cash or $2000 award.")
    print(classify_review(text1, gpt, tokenizer, device, max_length=datasets[0].max_length))
