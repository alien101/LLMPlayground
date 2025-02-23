GPT_CONFIG_124M = {
    "vocab_size":50257, # vocabulary size used by BPE tokenizer
    "context":1024,
    "emb_dim":768,
    "n_head":12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False,
}

GPT_medium = {
    "vocab_size":50257, # vocabulary size used by BPE tokenizer
    "context":1024,
    "emb_dim":1024,
    "n_head":16,
    "n_layers":24,
    "drop_rate":0.1,
    "qkv_bias":False,
}

GPT_large = {
    "vocab_size":50257, # vocabulary size used by BPE tokenizer
    "context":1024,
    "emb_dim":1280,
    "n_head":20,
    "n_layers":36,
    "drop_rate":0.1,
    "qkv_bias":False,
}

GPT_xlarge = {
    "vocab_size":50257, # vocabulary size used by BPE tokenizer
    "context":1024,
    "emb_dim":1600,
    "n_head":25,
    "n_layers":48,
    "drop_rate":0.1,
    "qkv_bias":False,
}
