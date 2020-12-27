
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from MulT.modules.position_embedding import SinusoidalPositionalEmbedding


from modules.multihead_attention import MultiHeadAttention, MaskMultiHeadAttention



# class PoswiseFeedForwardNet(nn.Module):
#     def __init__(self, hidden_size, ff_size):
#         super(PoswiseFeedForwardNet, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=ff_size, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=ff_size, out_channels=hidden_size, kernel_size=1)
#         self.layer_norm = nn.LayerNorm(hidden_size)
#
#     def forward(self, inputs):
#         residual = inputs # inputs : [batch_size, len_q, d_model]
#         output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
#         output = self.conv2(output).transpose(1, 2)
#         return self.layer_norm(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.fc1(inputs))
        output = self.fc2(output)
        return self.layer_norm(output + residual)


class SelfEncoderLayer(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size):
        super(SelfEncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(hidden_size, head_size, n_heads)
        self.layer_norm_before = nn.LayerNorm(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pos_ffn = PoswiseFeedForwardNet(hidden_size, ff_size)

    def forward(self, enc_inputs, enc_self_attn_mask):
        residual = enc_inputs
        enc_inputs = self.layer_norm_before(enc_inputs)
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.layer_norm(enc_outputs + residual)
        enc_outputs = self.pos_ffn(enc_outputs)     # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(hidden_size, head_size, n_heads)
        self.dec_enc_attn = MultiHeadAttention(hidden_size, head_size, n_heads)
        self.self_layer_norm_before = nn.LayerNorm(hidden_size)
        self.self_layer_norm = nn.LayerNorm(hidden_size)
        self.dec_enc_layer_norm = nn.LayerNorm(hidden_size)
        self.pos_ffn = PoswiseFeedForwardNet(hidden_size, ff_size)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        residual = dec_inputs
        dec_inputs = self.self_layer_norm_before(dec_inputs)
        dec_outputs, dec_self_attn_weights = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.self_layer_norm(dec_outputs + residual)

        residual = dec_outputs
        dec_outputs, dec_enc_attn_weights = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.dec_enc_layer_norm(dec_outputs + residual)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn_weights, dec_enc_attn_weights


class MaskCrossEncoderLayer(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size):
        super(MaskCrossEncoderLayer, self).__init__()
        self.mi_cross_attn = MaskMultiHeadAttention(hidden_size, head_size, n_heads)
        self.mp_cross_attn = MaskMultiHeadAttention(hidden_size, head_size, n_heads)
        self.self_attn = MultiHeadAttention(hidden_size, head_size, n_heads)
        self.layer_norm_before = nn.LayerNorm(hidden_size)
        self.mi_layer_norm = nn.LayerNorm(hidden_size)
        self.mp_layer_norm = nn.LayerNorm(hidden_size)
        self.self_layer_norm = nn.LayerNorm(hidden_size)
        self.pos_ffn = PoswiseFeedForwardNet(hidden_size, ff_size)

    def forward(self, k_v_inputs, q_inputs, cross_attn_mask, mi_mask=None, mp_mask=None):
        residual = q_inputs
        q_inputs = self.layer_norm_before(q_inputs)
        k_v_inputs = self.layer_norm_before(k_v_inputs)
        mi_cross_outputs, mi_cross_attn_weights = self.mi_cross_attn(q_inputs, k_v_inputs, k_v_inputs,
                                                                     cross_attn_mask, mask_mask=mi_mask)
        mp_cross_outputs, mp_cross_attn_weights = self.mp_cross_attn(q_inputs, k_v_inputs, k_v_inputs,
                                                                     cross_attn_mask, mask_mask=mp_mask)
        mi_norm_outputs = self.mi_layer_norm(mi_cross_outputs + residual)
        mp_norm_outputs = self.mp_layer_norm(mp_cross_outputs + residual)

        cross_norm_outputs = mi_norm_outputs + mp_norm_outputs
        self_outputs, self_attn_weights = self.self_attn(cross_norm_outputs, cross_norm_outputs, cross_norm_outputs,
                                                         cross_attn_mask)   # cross_attn_mask?
        sell_norm_outputs = self.self_layer_norm(self_outputs + cross_norm_outputs)

        enc_outputs = self.pos_ffn(sell_norm_outputs)     # enc_outputs: [batch_size x len_q x d_model]

        return enc_outputs, [mi_cross_attn_weights, mp_cross_attn_weights, self_attn_weights]


class ConcatMaskCrossEncoderLayer(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size):
        super(ConcatMaskCrossEncoderLayer, self).__init__()
        self.mi_cross_attn = MaskMultiHeadAttention(hidden_size, head_size, n_heads)
        self.mp_cross_attn = MaskMultiHeadAttention(hidden_size, head_size, n_heads)
        self.self_attn = MultiHeadAttention(hidden_size*2, head_size*2, n_heads)
        self.layer_norm_before = nn.LayerNorm(hidden_size)
        self.mi_layer_norm = nn.LayerNorm(hidden_size)
        self.mp_layer_norm = nn.LayerNorm(hidden_size)
        self.self_layer_norm = nn.LayerNorm(hidden_size*2)
        self.pos_ffn = PoswiseFeedForwardNet(hidden_size*2, ff_size*2)

    def forward(self, k_v_inputs, q_inputs, cross_attn_mask, mi_mask=None, mp_mask=None):
        residual = q_inputs
        q_inputs = self.layer_norm_before(q_inputs)
        k_v_inputs = self.layer_norm_before(k_v_inputs)
        mi_cross_outputs, mi_cross_attn_weights = self.mi_cross_attn(q_inputs, k_v_inputs, k_v_inputs,
                                                                     cross_attn_mask, mask_mask=mi_mask)
        mp_cross_outputs, mp_cross_attn_weights = self.mp_cross_attn(q_inputs, k_v_inputs, k_v_inputs,
                                                                     cross_attn_mask, mask_mask=mp_mask)
        mi_norm_outputs = self.mi_layer_norm(mi_cross_outputs + residual)
        mp_norm_outputs = self.mp_layer_norm(mp_cross_outputs + residual)

        cross_norm_outputs = torch.cat([mi_norm_outputs, mp_norm_outputs], dim=-1)
        self_outputs, self_attn_weights = self.self_attn(cross_norm_outputs, cross_norm_outputs, cross_norm_outputs,
                                                         cross_attn_mask)   # cross_attn_mask?
        sell_norm_outputs = self.self_layer_norm(self_outputs + cross_norm_outputs)

        enc_outputs = self.pos_ffn(sell_norm_outputs)     # enc_outputs: [batch_size x len_q x d_model]

        return enc_outputs, [mi_cross_attn_weights, mp_cross_attn_weights, self_attn_weights]


class SelfEncoder(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size, n_layers):
        super(SelfEncoder, self).__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_scale = np.sqrt(hidden_size)
        self.pos_emb = SinusoidalPositionalEmbedding(hidden_size)
        self.layers = nn.ModuleList([SelfEncoderLayer(hidden_size, head_size, n_heads, ff_size) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, enc_inputs, enc_self_attn_mask=None):  # enc_inputs : [batch_size x source_len x input_size]
        enc_inputs = self.hidden_scale * enc_inputs
        enc_outputs = enc_inputs + self.pos_emb(enc_inputs[:,:,0])

        all_enc_self_attn_weights = []
        for layer in self.layers:
            enc_outputs, enc_self_attn_weights = layer(enc_outputs, enc_self_attn_mask)
            all_enc_self_attn_weights.append(enc_self_attn_weights)

        enc_outputs = self.layer_norm(enc_outputs)

        return enc_outputs, all_enc_self_attn_weights


class Decoder(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size, n_layers):
        super(Decoder, self).__init__()
        self.hidden_scale = np.sqrt(hidden_size)
        self.pos_emb = SinusoidalPositionalEmbedding(hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(hidden_size, head_size, n_heads, ff_size) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask): # dec_inputs : [batch_size x target_len]
        dec_inputs = self.hidden_scale * dec_inputs
        enc_outputs = self.hidden_scale * enc_outputs
        dec_outputs = dec_inputs + self.pos_emb(dec_inputs[:, :, 0])
        enc_outputs = enc_outputs + self.pos_emb(enc_outputs[:,:,0])

        all_dec_self_attn_weights, all_dec_enc_attn_weights = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn_weights, dec_enc_attn_weights = layer(dec_outputs, enc_outputs,
                                                                             dec_self_attn_mask, dec_enc_attn_mask)
            all_dec_self_attn_weights.append(dec_self_attn_weights)
            all_dec_enc_attn_weights.append(dec_enc_attn_weights)

        dec_outputs = self.layer_norm(dec_outputs)

        return dec_outputs, all_dec_self_attn_weights, all_dec_enc_attn_weights


class MaskCrossEncoder(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size, n_layers):
        super(MaskCrossEncoder, self).__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_scale = np.sqrt(hidden_size)
        self.pos_emb = SinusoidalPositionalEmbedding(hidden_size)
        self.layers = nn.ModuleList(
            [MaskCrossEncoderLayer(hidden_size, head_size, n_heads, ff_size) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, k_v_inputs, q_inputs, enc_cross_attn_mask, mi_mask=None, mp_mask=None):
        k_v_inputs = self.hidden_scale * k_v_inputs
        q_inputs = self.hidden_scale * q_inputs
        k_v_inputs = k_v_inputs + self.pos_emb(k_v_inputs[:,:,0])
        q_inputs = q_inputs + self.pos_emb(q_inputs[:,:,0])

        all_mi_cross_attn_weights = []
        all_mp_cross_attn_weights = []
        all_self_attn_weights = []

        for layer in self.layers:
            enc_outputs, cross_attn_list = layer(k_v_inputs, q_inputs, enc_cross_attn_mask, mi_mask, mp_mask)
            mi_cross_attn_weights, mp_cross_attn_weights, self_attn_weights = cross_attn_list
            all_mi_cross_attn_weights.append(mi_cross_attn_weights)
            all_mp_cross_attn_weights.append(mp_cross_attn_weights)
            all_self_attn_weights.append(self_attn_weights)

        # enc_outputs = self.layer_norm(enc_outputs)

        return enc_outputs, [all_mi_cross_attn_weights, all_mp_cross_attn_weights, all_self_attn_weights]


class ConcatMaskCrossEncoder(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size, n_layers):
        super(ConcatMaskCrossEncoder, self).__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_scale = np.sqrt(hidden_size)
        self.pos_emb = SinusoidalPositionalEmbedding(hidden_size)
        self.layers = nn.ModuleList(
            [ConcatMaskCrossEncoderLayer(hidden_size, head_size, n_heads, ff_size) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, k_v_inputs, q_inputs, enc_cross_attn_mask, mi_mask=None, mp_mask=None):
        k_v_inputs = self.hidden_scale * k_v_inputs
        q_inputs = self.hidden_scale * q_inputs
        k_v_inputs = k_v_inputs + self.pos_emb(k_v_inputs[:,:,0])
        q_inputs = q_inputs + self.pos_emb(q_inputs[:,:,0])

        all_mi_cross_attn_weights = []
        all_mp_cross_attn_weights = []
        all_self_attn_weights = []

        for layer in self.layers:
            enc_outputs, cross_attn_list = layer(k_v_inputs, q_inputs, enc_cross_attn_mask, mi_mask, mp_mask)
            mi_cross_attn_weights, mp_cross_attn_weights, self_attn_weights = cross_attn_list
            all_mi_cross_attn_weights.append(mi_cross_attn_weights)
            all_mp_cross_attn_weights.append(mp_cross_attn_weights)
            all_self_attn_weights.append(self_attn_weights)

        # enc_outputs = self.layer_norm(enc_outputs)

        return enc_outputs, [all_mi_cross_attn_weights, all_mp_cross_attn_weights, all_self_attn_weights]
