#import urllib.request

#url = (
#    "https://raw.githubusercontent.com/rasbt/"
#    "LLMs-from-scratch/main/ch05/"
#    "01_main-chapter-code/gpt_download.py"
#)
#filename= url.split('/')[-1]
#urllib.request.urlretrieve(url, filename)

from gpt_download import download_and_load_gpt2
from architecture.config import *
from module.GPT.model import *
import torch
import numpy as np
from train_simple import generate
import tiktoken

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


if __name__ == "__main__":
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    gpt = GPT(GPT_CONFIG_124M)
    load_weights_into_gpt(gpt, params)

    start_context="Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    gpt.cuda()
    torch.manual_seed(123)
    token_ids = generate(model=gpt,
                        idx=text_to_token(start_context, tokenizer).to('cuda'),
                        max_new_tokens=25,
                        context_size=GPT_CONFIG_124M["context"],
                        temperature=1.5,
                        topk=50
                        )   
    text = token_to_text(token_ids, tokenizer) 
    print(text)
    pass