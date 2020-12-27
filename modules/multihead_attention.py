
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, attn_mask):
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)    # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9)    # Fills elements of self tensor with value where mask is one.
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        return context, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.head_size = head_size
        self.n_heads = n_heads
        self.W_q = nn.Linear(hidden_size, head_size * n_heads)
        self.W_k = nn.Linear(hidden_size, head_size * n_heads)
        self.W_v = nn.Linear(hidden_size, head_size * n_heads)
        self.scale_dot_product = ScaledDotProductAttention()
        self.linear = nn.Linear(n_heads * head_size, hidden_size)

    def forward(self, q, k, v, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        batch_size = q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_q(q).view(batch_size, -1, self.n_heads, self.head_size).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_k(k).view(batch_size, -1, self.n_heads, self.head_size).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_v(v).view(batch_size, -1, self.n_heads, self.head_size).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn_weights = self.scale_dot_product(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_size) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return output, attn_weights     # output: [batch_size x len_q x d_model]


class MaskMultiHeadAttention(MultiHeadAttention):
    def __init__(self, hidden_size, head_size, n_heads):
        super(MaskMultiHeadAttention, self).__init__(hidden_size, head_size, n_heads)

    def forward(self, q, k, v, attn_mask, mask_mask=None):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        batch_size = q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_q(q).view(batch_size, -1, self.n_heads, self.head_size).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_k(k).view(batch_size, -1, self.n_heads, self.head_size).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_v(v).view(batch_size, -1, self.n_heads, self.head_size).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        if mask_mask is not None:
            if len(mask_mask.shape) == 3:
                attn_mask = attn_mask | mask_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            else:
                attn_mask = attn_mask | mask_mask
        context, attn_weights = self.scale_dot_product(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_size) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return output, attn_weights     # output: [batch_size x len_q x d_model]