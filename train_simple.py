from loss.loss import *
from module.GPT.model import *
from module.uitls import *
import tiktoken
import torch
from architecture.config import *
from datamodule.dataloader import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datamodule.dataset import *
import tiktoken

def train_model_simple(model, train_loader, valid_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, 
                       temperature=1.4, topk=25):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input, target in train_loader:
            optimizer.zero_grad()
            loss = calc_cross_entroply_loss(input=input,
                                            target=target,
                                            model=model,
                                            device=device)
            loss.backward()
            optimizer.step()
            tokens_seen+=input.numel()
            global_step += 1
            # frequency of validation update
            if global_step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(model, 
                                                        train_loader, valid_loader, 
                                                        device, 
                                                        eval_iter)
                train_losses.append(train_loss.cpu().numpy())
                val_losses.append(valid_loss.cpu().numpy())
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch{epoch + 1} (step {global_step:06d}): "
                      f"Train loss {train_loss:.3f} "
                      f"Val loss {valid_loss:.3f}"
                )
        generate_and_print_sample(model, tokenizer, device, start_context, 
                                  temperature=temperature, k=topk)

    return train_losses, val_losses, track_tokens_seen

def split_data():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    train_ratio = 0.1
    split_idx = int(len(raw_text)*train_ratio)

    valid_data = raw_text[:split_idx]
    train_data = raw_text[split_idx:]

    train_loader = create_dataloader(
        txt=train_data,
        batch_size=2,
        max_len=256,
        stride=128,
    )
    valid_loader = create_dataloader(
        txt=valid_data,
        batch_size=2,
        max_len=256,
        stride=128,
        drop_last=False,
        shuffle=False
    )
    return train_loader, valid_loader

def plot_losses(epochs_seen, toekns_senn, train_losses, val_losses):
  fig, ax1 = plt.subplots(figsize=(5,3))
  ax1.plot(epochs_seen, train_losses, label="training loss")
  ax1.plot(epochs_seen, val_losses, linestyle="-.", label="validation loss")
  ax1.set_xlabel("Epochs")
  ax1.set_ylabel("Loss")
  ax1.legend(loc="upper right")
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax2 = ax1.twiny()
  ax2.plot(toekns_senn, train_losses, alpha=0)
  ax2.set_xlabel("Token seen")
  fig.tight_layout()
  plt.show()

if __name__ == "__main__":
    torch.manual_seed(123)
    
    # train generative text
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPT(GPT_CONFIG_124M)
    device = "cuda"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs=10
    train_loader, val_loader = split_data()
    train_losses, val_losses, token_seen = train_model_simple(model,
                                                              train_loader,
                                                              val_loader,
                                                              optimizer,
                                                              device,
                                                              num_epochs,
                                                              eval_freq=5,
                                                              eval_iter=5,
                                                              start_context="Every effort moves you",
                                                              tokenizer=tokenizer)
    torch.save({"model_state_dict":model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                }, "temp.pth")
    ckpt = torch.load("temp.pth", map_location=device)
    


    pass