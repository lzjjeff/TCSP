
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from MulT.modules.position_embedding import SinusoidalPositionalEmbedding


from transformer.modules.multihead_attention import MultiHeadAttention, MultiHeadAttentionFromPretrained



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=ff_size, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_size, out_channels=hidden_size, kernel_size=1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class SelfEncoderLayer(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size):
        super(SelfEncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(hidden_size, head_size, n_heads)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pos_ffn = PoswiseFeedForwardNet(hidden_size, ff_size)

    def forward(self, enc_inputs, enc_self_attn_mask):
        residual = enc_inputs
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.layer_norm(enc_outputs + residual)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn_weights


class CrossEncoderLayer(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size):
        super(CrossEncoderLayer, self).__init__()
        self.cross_attn_fix = MultiHeadAttentionFromPretrained(hidden_size, head_size, n_heads)
        self.cross_attn_fle = MultiHeadAttentionFromPretrained(hidden_size, head_size, n_heads)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pos_ffn = PoswiseFeedForwardNet(hidden_size, ff_size)

    def forward(self, src_inputs, tgt_inputs, cross_attn_mask, cross_attn_weights=None):
        residual = tgt_inputs
        cross_outputs_fix, cross_attn_weights_fix = self.cross_attn_fix(tgt_inputs, src_inputs, src_inputs,
                                                                        cross_attn_mask, attn_weights=cross_attn_weights)
        cross_outputs_fle, cross_attn_weights_fle = self.cross_attn_fle(tgt_inputs, src_inputs, src_inputs,
                                                                        cross_attn_mask)
        enc_outputs = self.layer_norm(cross_outputs_fix + cross_outputs_fle + residual)
        enc_outputs = self.pos_ffn(enc_outputs)     # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, [cross_attn_weights_fix, cross_attn_weights_fle]


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(hidden_size, head_size, n_heads)
        self.dec_enc_attn = MultiHeadAttention(hidden_size, head_size, n_heads)
        self.self_layer_norm = nn.LayerNorm(hidden_size)
        self.dec_enc_layer_norm = nn.LayerNorm(hidden_size)
        self.pos_ffn = PoswiseFeedForwardNet(hidden_size, ff_size)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        residual = dec_inputs
        dec_outputs, dec_self_attn_weights = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.self_layer_norm(dec_outputs + residual)

        residual = dec_outputs
        dec_outputs, dec_enc_attn_weights = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.dec_enc_layer_norm(dec_outputs + residual)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn_weights, dec_enc_attn_weights


class SelfEncoder(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size, n_layers):
        super(SelfEncoder, self).__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_scale = np.sqrt(hidden_size)
        self.pos_emb = SinusoidalPositionalEmbedding(hidden_size)
        self.layers = nn.ModuleList([SelfEncoderLayer(hidden_size, head_size, n_heads, ff_size) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, enc_inputs, enc_self_attn_mask): # enc_inputs : [batch_size x source_len x input_size]
        enc_inputs = self.hidden_scale * enc_inputs
        enc_outputs = enc_inputs + self.pos_emb(enc_inputs[:,:,0])

        all_enc_self_attn_weights = []
        for layer in self.layers:
            enc_outputs, enc_self_attn_weights = layer(enc_outputs, enc_self_attn_mask)
            all_enc_self_attn_weights.append(enc_self_attn_weights)

        enc_outputs = self.layer_norm(enc_outputs)

        return enc_outputs, all_enc_self_attn_weights


class CrossEncoder(nn.Module):
    def __init__(self, hidden_size, head_size, n_heads, ff_size, n_layers):
        super(CrossEncoder, self).__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_scale = np.sqrt(hidden_size)
        self.pos_emb = SinusoidalPositionalEmbedding(hidden_size)
        self.layers = nn.ModuleList(
            [CrossEncoderLayer(hidden_size, head_size, n_heads, ff_size) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, src_inputs, tgt_inputs, enc_cross_attn_mask, all_cross_attn_weights=None, shared_list=None):
        src_inputs = self.hidden_scale * src_inputs
        tgt_inputs = self.hidden_scale * tgt_inputs
        enc_outputs = src_inputs + self.pos_emb(src_inputs[:,:,0])
        tgt_inputs = tgt_inputs + self.pos_emb(tgt_inputs[:,:,0])

        all_enc_cross_attn_weights = []
        if all_cross_attn_weights is not None:
            if shared_list is not None:
                for layer, shared_idx in zip(self.layers, shared_list):
                    if shared_idx > -1:
                        enc_outputs, enc_cross_attn_weights = layer(src_inputs, tgt_inputs, enc_cross_attn_mask,
                                                                    all_cross_attn_weights[shared_idx])
                    else:
                        enc_outputs, enc_cross_attn_weights = layer(src_inputs, tgt_inputs, enc_cross_attn_mask)
                    all_enc_cross_attn_weights.append(enc_cross_attn_weights)
            else:
                if len(all_cross_attn_weights) == self.n_layers:
                    for layer, cross_attn_weights in zip(self.layers, all_cross_attn_weights):
                        enc_outputs, enc_cross_attn_weights = layer(src_inputs, tgt_inputs, enc_cross_attn_mask,
                                                                    cross_attn_weights)
                        all_enc_cross_attn_weights.append(enc_cross_attn_weights)
                else:
                    for layer in self.layers:
                        enc_outputs, enc_cross_attn_weights = layer(src_inputs, tgt_inputs, enc_cross_attn_mask,
                                                                    all_cross_attn_weights[0])
                        all_enc_cross_attn_weights.append(enc_cross_attn_weights)
        else:
            for layer in self.layers:
                enc_outputs, enc_cross_attn_weights = layer(src_inputs, tgt_inputs, enc_cross_attn_mask)
                all_enc_cross_attn_weights.append(enc_cross_attn_weights)

        enc_outputs = self.layer_norm(enc_outputs)

        return enc_outputs, all_enc_cross_attn_weights


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

