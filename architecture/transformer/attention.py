import torch
from torch import nn, Tensor

# Unlearnable SelfAttention
def simpleSelfAttention(V: Tensor) -> Tensor:
    """
        Args: 
            V (Tensor): 1D text embedded vector
                        Shape of (1,N)
        Returns:
            Tensor: 1D final weighted embedding
                    Shape of (1, N)
    """
    V = V.view(1, -1) 
    # weights to reweight V
    W = torch.matmul(V.T, V)
    # normaliza, so the sum of each row is 1 
    W_norm = W / torch.sum(W, dim=1, keepdim=True)
    # each row os the final weighted embedding of the word with context
    Y = torch.matmul(W_norm, V.float()) 
    
    return Y

# Learnable SelfAttention
class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.dim = d_out
        self.W_k = nn.Parameter(torch.rand(d_in, d_out))
        self.W_v = nn.Parameter(torch.rand(d_in, d_out))
        self.W_q = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # x is the rich-embedded text with tokenizer convert 
        # to embeddings plus positional embedding
        q = torch.matmul(x, self.W_q) # aka. q = x @ self.W_q
        k = torch.matmul(x, self.W_k) # aka. k = x @ self.W_k
        v = torch.matmul(x, self.W_v) # aka. v = x @ self.W_v

        attn_score = torch.matmul(q, torch.transpose(k, -2, -1))
        weighted_attn_score = torch.softmax(attn_score/ self.dim**0.5, dim=-1)
        
        return weighted_attn_score, v   
    
# Learnable SelfAttention
class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.dim = d_out
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)
        self.W_q = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x):
        # x is the rich-embedded text with tokenizer convert 
        # to embeddings plus positional embedding
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attn_score = torch.matmul(q, torch.transpose(k, -2, -1))
        weighted_attn_score = torch.softmax(attn_score/ self.dim**0.5, dim=-1)
        
        return weighted_attn_score, v

# Final implementation
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.dim = d_out
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)
        self.W_q = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x):
        # x is the rich-embedded text with tokenizer convert 
        # to embeddings plus positional embedding
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attn_score = torch.matmul(q, torch.transpose(k, -2, -1))
        weighted_attn_score = torch.softmax(attn_score/ self.dim**0.5, dim=-1)
        
        return torch.matmul(weighted_attn_score, v)

# Assuming context length equals to number of tokens
class CausalAttentionV1(SelfAttentionV2):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__(d_in, d_out, bias)
    
    def forward(self, x):
        norm_attn_weights, v = super().forward(x)
        context_length = norm_attn_weights.shape[1]
        mask_simple = torch.tril(torch.ones(context_length, context_length))
        # set masked as 0
        masked_attn_weights = mask_simple*norm_attn_weights
        masked_norm_attn_weights = masked_attn_weights/ torch.sum(masked_attn_weights, dim=-1, keepdim=True)

        return masked_norm_attn_weights, v
    
# Use softmax to normalize for efficiency 
class CausalAttentionV2(SelfAttentionV2):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__(d_in, d_out, bias)

    def forward(self, x):
        norm_attn_weights, v = super().forward(x)
        context_length = norm_attn_weights.shape[1]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        # set masked as -inf
        masked_attn_weights = norm_attn_weights.masked_fill(mask.bool(), -torch.inf)
        masked_norm_attn_weights = torch.softmax(masked_attn_weights/self.dim**0.5, dim=-1)

        return masked_norm_attn_weights, v

# Final version of implementatopn, number of tokens is less or equals to context length        
class CausalAttention(CausalAttentionV2):
    def __init__(self, d_in, d_out, context_length, dropout=0, bias=False):
        super().__init__(d_in, d_out, bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        _, num_token, _ = x.shape
        norm_attn_weights, v = super().forward(x)
        # set masked as -inf
        masked_attn_weights = norm_attn_weights.masked_fill(
                                    self.mask.bool()[:num_token, :num_token],
                                    -torch.inf
                                )
        masked_norm_attn_weights = torch.softmax(masked_attn_weights/self.dim**0.5, dim=-1)
        masked_norm_attn_weights = self.dropout(masked_norm_attn_weights)
        x = torch.matmul(masked_norm_attn_weights, v)
        
        return x
    
# Implementation based on online source
class MultiHeadAttentionScratch(nn.Module):
    """
    Implementation of attention based on this resource 
    https://towardsdatascience.com/all-you-need-to-know-about-attention-and-transformers-in-depth-understanding-part-1-552f0b41d021
    """

    def __init__(self, embedding_size:int, num_heads: int):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        # The dimension after spliting heads
        self.dk = embedding_size // num_heads
        # Trainable parameters for K, Q, V that is not in self attention 
        self.W_v = nn.Linear(embedding_size, embedding_size)
        self.W_q = nn.Linear(embedding_size, embedding_size)
        self.W_k = nn.Linear(embedding_size, embedding_size)

    def splitHeads(self, x: Tensor) -> Tensor:
        """
        This function splits up the attention into multiple layers to improve in 
        identifying the relevancy among words

        Args:
            x (Tensor): with shape (batch size (B), senquence length (L), embedding size (N))

        Returns:
            Tensor: a attention layer splited into num_heads attention layer.
            Shape of (B, self.num_heads, L, self.dk)
        """
        batch_size, sequence_length, _ = x.size()

        return x.view(batch_size, sequence_length, self.num_heads, self.dk).transpose(1, 2)

    def mergeHeads(self, multi_head: Tensor) -> Tensor:
        """
        This function combines the attention output from all the heads

        Args:
            multi_head (Tensor): the multi-head attention output
        Returns:
            Tensor: combined multi-head attention output
        """
        batch_size, _, seq_length, _ = multi_head.size()
        
        #return multi_head.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_size)
        return multi_head.transpose(1, 2).view(batch_size, seq_length, self.embedding_size)

    def attention(self, K, Q, V):
        """
        Implementation of the calculation for final weighted embedding in self attention

        Args:
            K (Tensor): Keys, first occurence of initial embedded vector. 
                        Shape: (B, self.num_heads, L, self.dk)
            Q (Tensor): Query, second occurence of initial embedded vector, which operates dot 
                        product with K
                        Shape: (B, self.num_heads, L, self.dk)
            V (Tensor): Values, third occurance of initial embedded vector. which is multiplied
                        to obtain final weighted embedded vector
                        Shape: (B, self.num_heads, L, self.dk)
        Returns:
            Y (Tensor): final weighted embedded vector
                        Shape: (B, self.num_heads, L)
        """
        ## This is the same as W = torch.matmul(V.T, V)  in selfAttention, in respect of having a 
        #  splited attention
        W = torch.matmul(Q, K.transpose(-2, -1))
        ## Scaling. This is to prevent overflow in weight leading to vanish gradient in training.
        #   Square root is applied as 
        #       dk = 1 -> Var(x) = A 
        #       dk = 2 -> Var(2x) = 4A
        #    ...dk = n -> Var(nx) = [(n)^2]A
        #   Hence, we will divied W by sqrt(dk) to keep the distribution stable    
        W_norm = W / torch.sqrt(self.dk) # aka attention scores
        
        attn_prob = torch.softmax(W_norm, dim=-1)
        Y = torch.matmul(attn_prob, V)
        
        return Y

    def forward(self, K, Q, V):
        Q = self.splitHeads(self.W_q(Q))
        K = self.splitHeads(self.W_k(K))
        V = self.splitHeads(self.W_v(V))

        multiAttention = self.attention(K, Q, V)
        mergedAttention = self.mergeHeads(multiAttention)
        return mergedAttention

# Serial implementation
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, num_heads, d_in, d_out, context_lenght, dropout=0, bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in=d_in, 
                             d_out=d_out,
                             context_length=context_lenght,
                             dropout=dropout,
                             bias=bias
            ) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# Parallel implementation
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_in, d_out, context_length, dropout=0.1, bias=False):
        super().__init__()
        assert d_out % num_heads == 0

        self.num_heads=num_heads        
        self.d_in=d_in
        self.d_out=d_out
        self.context_length=context_length
        self.dropout=nn.Dropout(dropout)
        
        self.head_dim = d_out // num_heads
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)
        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        # Optional, mostly used in LLM
        self.combine_head = nn.Linear(d_out, d_out, bias=bias)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        # shape of (batch_size, num_tokens, d_out)
        k = self.W_k(x)
        q = self.W_q(x)
        v = self.W_v(x)
        # reshape to shape of (batch_size, num_tokens, num_heads, head_dim)
        k = k.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        q = q.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        v = v.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        # tranpose to shape of (batch_size, num_heads, num_tokens, head_dim)
        k = torch.transpose(k, 1, 2)
        q = torch.transpose(q, 1, 2)
        v = torch.transpose(v, 1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(2, 3))
        # masking
        attn_scores = attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # normalizing
        attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)
        # transpose to the shape of (batch_size, num_tokens, num_heads, head_dim)
        context_vector = torch.matmul(attn_weights, v).transpose(1, 2)
        # Tensor needs to be contiguous to use torch.view(), torch.transpose creates non-contiguous tensors
        context_vector = context_vector.contiguous().view(batch_size, num_tokens, self.d_out)
        # Optional, mostly used in LLM
        context_vector = self.combine_head(context_vector)
        
        return context_vector

if __name__ == "__main__":
    torch.manual_seed(42)
    input = torch.randn(2, 6, 3)
    batch_size, context_length, d_in = input.shape
    d_out = 2
    
    #attentionV1 = SelfAttentionV1(d_in=d_in, d_out=d_out)
    #attentionV2 = SelfAttentionV2(d_in=d_in, d_out=d_out)
    #score2, v = attentionV2(input)
    #score1, v = attentionV1(input)
    #
    #assert torch.all(score1.flatten() != score2.flatten())
#
    ## Exercise 3.1: Assign weights on V2 to V1
    #attentionV1.W_k = nn.Parameter(attentionV2.state_dict()["W_k.weight"].T)#, 1, 0))
    #attentionV1.W_q = nn.Parameter(attentionV2.state_dict()["W_q.weight"].T)#, 1, 0))
    #attentionV1.W_v = nn.Parameter(attentionV2.state_dict()["W_v.weight"].T)#, 1, 0))
    #score2, v = attentionV2(input)
    #score1, v = attentionV1(input)
#
    #assert torch.all(score1.flatten() == score2.flatten())    
    
    # Ch.3.5
    #attention = CausalAttention(d_in=d_in, 
    #                            d_out=d_out, 
    #                            context_length=context_length, 
    #                            dropout=0)

    # Ch.3.6
    #attention = MultiHeadAttentionWrapper(num_heads=4,
    #                                      d_in=d_in, 
    #                                      d_out=d_out,
    #                                      context_lenght=context_length)
    #attention = MultiHeadAttention(num_heads=4, 
    #                               d_in=d_in, 
    #                               d_out=d_out*4, 
    #                               context_length=context_length)

    # Exercise 3.2: returning 2D embedding vectors
    #d_out = 1
    #attention = MultiHeadAttentionWrapper(num_heads=2,
    #                                      d_in=d_in, 
    #                                      d_out=d_out,
    #                                      context_lenght=context_length)

    #output = attention(input)
    #print(output.shape)

    #Exercise 3.3: Initializing GPT-2 size attention modules
    #gpt2_attn = MultiHeadAttention(num_heads=12,
    #                               d_in=768*12,
    #                               d_out=768*12,
    #                               context_length=1024,
    #                               bias=True)
    #print(sum(p.numel() for p in gpt2_attn.parameters()))



    