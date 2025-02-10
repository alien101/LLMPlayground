"""
Implementation of attention based on this resource 
https://towardsdatascience.com/all-you-need-to-know-about-attention-and-transformers-in-depth-understanding-part-1-552f0b41d021
"""

import torch
from torch import nn, Tensor

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
    
class SelfAttention(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.dim = embedding_size
        self.W_k = nn.Linear(embedding_size, embedding_size)
        self.W_v = nn.Linear(embedding_size, embedding_size)
        self.W_q = nn.Linear(embedding_size, embedding_size)

    def forward(self, context_vectors):
        # context_vectors is the rich-embedded text with tokenizer convert 
        # to embeddings plus positional embedding
        q = self.W_q(context_vectors)
        k = self.W_k(context_vectors)
        v = self.W_v(context_vectors)

        attention_score = torch.matmul(q, torch.transpose(k, -2, -1))
        weighted_attention_score = torch.softmax(attention_score/ self.dim**0.5, dim=-1)
        
        return torch.matmul(weighted_attention_score, v)



class MultiHeadAttention(nn.Module):
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
        
        attention_prob = torch.softmax(W_norm, dim=-1)
        Y = torch.matmul(attention_prob, V)
        
        return Y

    def forward(self, K, Q, V):
        Q = self.splitHeads(self.W_q(Q))
        K = self.splitHeads(self.W_k(K))
        V = self.splitHeads(self.W_v(V))

        multiAttention = self.attention(K, Q, V)
        mergedAttention = self.mergeHeads(multiAttention)
        return mergedAttention



if __name__ == "__main__":
    input = torch.randn(4, 6, 9)
    attention = SelfAttention(embedding_size=9)
    score = attention(input)

