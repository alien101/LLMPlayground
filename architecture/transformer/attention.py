"""
Implementation of attention based on this resource 
https://towardsdatascience.com/all-you-need-to-know-about-attention-and-transformers-in-depth-understanding-part-1-552f0b41d021
"""

import torch
from torch import nn, Tensor

def selfAttention(V: Tensor) -> Tensor:
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

    def split_heads(self, x):
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

        pass















{
"2024_08_21__20240806-2198A-010-UBL-H-LIVER-CL_4-PLS_22.tif": "20240905-2198A-010-UBL-H-LIVER-CL_4-PLS_22_PASD.tif",
"2024_08_21__20240806-2199A-010-UBL-H-LIVER-CL_4-PLS_22.tif": "20240905-2199A-010-UBL-H-LIVER-CL_4-PLS_22_PASD.tif",
"20240703-0961E-010-UBL-H-LIVER-CL4-PLS_22_RESCAN.tif":"20240703-0961E-010-UBL-H-LIVER-CL_4-PLS_22_PASD.tif",
"20240703-1461F-010-UBL-H-LIVER-CL4-PLS_22.tif":"20240703-1461F-010-UBL-H-LIVER-CL4-PLS_22_PASD_RESCAN.tif",
"20240703-01988-010-UBL-H-LIVER-CL4-PLS_22.tif":"20240703-01988-010-UBL-H-LIVER-CL_4-PLS_22_PASD.tif",
"20240703-3261H-010-UBL-H-LIVER-CL4-PLS_22.tif":"20240703-3261H-010-UBL-H-LIVER-CL_4-PLS_22_PASD.tif",
"20240703-6031D-010-UBL-H-LIVER-CL4-PLS_22.tif":"20240703-6031D-010-UBL-H-LIVER-CL_4-PLS_22_PASD.tif",
"20240703-6608B-010-UBL-H-LIVER-CL4-PLS_22.tif":"20240703-6608B-010-UBL-H-LIVER-CL_4-PLS_22_PASD.tif",
"20240703-7429E-010-UBL-H-LIVER-CL4-PLS_22.tif":"20240703-7429E-010-UBL-H-LIVER-CL_4-PLS_22_PASD.tif",
"20240703-8530M-010-UBL-H-LIVER-CL4-PLS_22.tif":"20240703-8530M-010-UBL-H-LIVER-CL_4-PLS_22_PASD.tif",
"20240703-14039-010-UBL-H-LIVER-CL4-PLS_22.tif":"20240703-14039-010-UBL-H-LIVER-CL_4-PLS_22_PASD.tif",
"20240806-3001N-010-UBL-H-LIVER-CL4-PLS_22.tif":"20240905-3001N-010-UBL-H-LIVER-CL_4-PLS_22_PASD.tif",
}