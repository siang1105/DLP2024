import torch.nn as nn
import torch
import math
import einops
import torch.nn.functional as F

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # Defines linear transformation layer Query, Key, Value
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        
        # Define output layer
        self.out_linear = nn.Linear(dim, dim)
        
        # Define Dropout layer
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, num_tokens, dim = x.shape
        
        # Projection Query, Key, Value
        q = self.q_linear(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose for dot product attention calculations
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_tokens, head_dim]
        k = k.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_tokens, head_dim]
        v = v.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_tokens, head_dim]
        
        # Compute scaling dot product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply Dropout
        attn_weights = self.attn_drop(attn_weights)
        
        # weighted average
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, num_tokens, head_dim]
        
        # Merge results from multiple heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, num_tokens, self.dim)
        
        # The last layer of linear transformation
        output = self.out_linear(attn_output)
        
        return output



class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    