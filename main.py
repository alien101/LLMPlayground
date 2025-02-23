from architecture.config import *
from architecture.transformer.transformer import *
from module.dummyGPT.model import *
from module.GPT.model import *

if __name__ == "__main__":
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = "this is a line \n this is another line 2".split('\n')
    batch = [torch.tensor(tokenizer.encode(txt)) for txt in batch]
    input = torch.stack(batch, dim=0)
    #
    #torch.manual_seed(123)
    #model = DummyGPTModel(GPT_CONFIG_124M)
    #logits = model(input)

    #x = torch.rand(2, 4, 768)
    #block = TransformerBlock(   GPT_CONFIG_124M)
    #output = block(x)
    
    model = GPT(GPT_CONFIG_124M)
    output = model(input)
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    output = inference(model=model, 
                       idx=encoded_tensor, 
                       max_new_tokens=6,
                       context_size=GPT_CONFIG_124M["context"])
    output_string = tokenizer.decode(output.squeeze().tolist())
    #total_params = sum([p.numel() for p in model.parameters()])
    #print(f"input shape: {input.shape} \n output shape: {output.shape}")
    
    # Excercise 4.1: NUmber of parameters in feed foward and attention
    #ff = FeedForward(GPT_CONFIG_124M)
    #mha = MultiHeadAttention(num_heads=GPT_CONFIG_124M["n_head"], 
    #                         d_in=GPT_CONFIG_124M["emb_dim"],
    #                         d_out=GPT_CONFIG_124M["emb_dim"],
    #                         dropout=GPT_CONFIG_124M["drop_rate"],
    #                         context_length=GPT_CONFIG_124M["context"]
    #                         )
    #print(f"FeedForward: {sum(p.numel() for p in ff.parameters())}")                             
    #print(f"MultiHeadAttn: {sum(p.numel() for p in mha.parameters())}")
    # >>> FeedForward: 4722432
    # >>> MultiHeadAttn: 2359296

    #model_size_MB = total_params * 4 /(1024 ** 2)
    #print(f"Model size: {model_size_MB} MB")
    # >>> Model size: 621.796875 MB

    # Exercise 4.2: Initializing lager GPT models
    #gpt_m = GPT(GPT_medium)
    #gpt_l = GPT(GPT_large)
    #gpt_xl = GPT(GPT_xlarge)
    #for size, gpt in zip(["medium", "large", "xlarge"], [gpt_m, gpt_l, gpt_xl]):
    #    print(f"num parameters of GPT {size}: {sum([p.numel() for p in gpt.parameters()])}")
    # >>> num parameters of GPT medium: 406188032
    # >>> num parameters of GPT large: 838174720
    # >>> num parameters of GPT xlarge: 1637715200
    pass