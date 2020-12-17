#
# code by Zijie Lin @jeff97
# Reference : https://github.com/graykode/nlp-tutorial/tree/master/4-2.Seq2Seq(Attention)
#
import numpy as np
import torch
import torch.nn as nn
from transformer.modules.transformer import SelfEncoder, CrossEncoder, Decoder


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q, _ = seq_q.size()
    batch_size, len_k, _ = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.sum(dim=-1).data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(seq.device)
    return subsequent_mask


class Translation(nn.Module):
    def __init__(self, config):
        super(Translation, self).__init__()
        self.config = config
        self.enc_proj = nn.Conv1d(in_channels=config["encoder_input_size"],
                                  out_channels=config["encoder_hidden_size"],
                                  kernel_size=1, padding=0, bias=False)
        self.dec_proj = nn.Conv1d(in_channels=config["decoder_output_size"],
                                  out_channels=config["decoder_hidden_size"],
                                  kernel_size=1, padding=0, bias=False)
        self.encoder = SelfEncoder(hidden_size=config["encoder_hidden_size"],
                                   head_size=config["encoder_head_size"],
                                   ff_size=config["ff_size"],
                                   n_heads=config["n_heads"],
                                   n_layers=config["encoder_n_layers"])
        self.decoder = Decoder(hidden_size=config["decoder_hidden_size"],
                               head_size=config["decoder_head_size"],
                               ff_size=config["ff_size"],
                               n_heads=config["n_heads"],
                               n_layers=config["decoder_n_layers"])
        self.projection = nn.Linear(config["decoder_hidden_size"], config["decoder_output_size"])

    def __get_attn_parameters(self):
        return [layer.dec_enc_attn.W_q.state_dict() for layer in self.decoder.layers], \
               [layer.dec_enc_attn.W_k.state_dict() for layer in self.decoder.layers]

    def forward(self, source, target):
        enc_inputs = self.enc_proj(source.permute(1, 2, 0).contiguous()).permute(2, 0, 1).contiguous()
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_outputs, all_enc_self_attn_weights = self.encoder(enc_inputs, enc_self_attn_mask)       # (b, enc_n, enc_h_d)

        # decode
        dec_inputs = self.dec_proj(target.permute(1, 2, 0).contiguous()).permute(2, 0, 1).contiguous()
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_outputs, all_dec_self_attn_weights, all_dec_enc_attn_weights = self.decoder(dec_inputs, enc_outputs,
                                                                                        dec_self_attn_mask,
                                                                                        dec_enc_attn_mask)
        dec_logits = self.projection(dec_outputs)

        return dec_logits, all_dec_enc_attn_weights


class DualAttentionRegressoin(nn.Module):
    def __init__(self, config):
        super(DualAttentionRegressoin, self).__init__()
        self.config = config
        self.fixed = config["fixed"]
        self.use_shared_list = config["use_shared_list"]
        self.shared_list = config["shared_list"]
        self.w_proj = nn.Conv1d(in_channels=config["w_size"],
                                out_channels=config["hidden_size"],
                                kernel_size=1, padding=0, bias=False)
        self.v_proj = nn.Conv1d(in_channels=config["v_size"],
                                out_channels=config["hidden_size"],
                                kernel_size=1, padding=0, bias=False)
        self.a_proj = nn.Conv1d(in_channels=config["a_size"],
                                out_channels=config["hidden_size"],
                                kernel_size=1, padding=0, bias=False)

        self.w_self_trm = SelfEncoder(hidden_size=config["hidden_size"],
                                      head_size=config["head_size"],
                                      ff_size=config["ff_size"],
                                      n_heads=config["n_heads"],
                                      n_layers=config["n_layers"])
        self.v2w_cross_tfm = CrossEncoder(hidden_size=config["hidden_size"],
                                          head_size=config["head_size"],
                                          ff_size=config["ff_size"],
                                          n_heads=config["n_heads"],
                                          n_layers=config["n_layers"])
        self.a2w_cross_trm = CrossEncoder(hidden_size=config["hidden_size"],
                                          head_size=config["head_size"],
                                          ff_size=config["ff_size"],
                                          n_heads=config["n_heads"],
                                          n_layers=config["n_layers"])
        self.v2w_self_trm = SelfEncoder(hidden_size=config["hidden_size"],
                                        head_size=config["head_size"],
                                        ff_size=config["ff_size"],
                                        n_heads=config["n_heads"],
                                        n_layers=config["n_layers"])
        self.a2w_self_trm = SelfEncoder(hidden_size=config["hidden_size"],
                                        head_size=config["head_size"],
                                        ff_size=config["ff_size"],
                                        n_heads=config["n_heads"],
                                        n_layers=config["n_layers"])
        self.fc = nn.Linear(config["hidden_size"]*2, 1)

    def __fix_attn_params(self, v_params, a_params):
        self.v_attn.fix_attn_params(v_params)
        self.a_attn.fix_attn_params(a_params)

    def forward(self, w, v, a, all_v_attn_weights=None, all_a_attn_weights=None):
        w_proj = self.w_proj(w.permute(1, 2, 0).contiguous()).permute(2, 0, 1).contiguous()
        v_proj = self.v_proj(v.permute(1, 2, 0).contiguous()).permute(2, 0, 1).contiguous()
        a_proj = self.a_proj(a.permute(1, 2, 0).contiguous()).permute(2, 0, 1).contiguous()

        # 备用
        w_self_attn_mask = get_attn_pad_mask(w_proj, w_proj)
        w_self_hs, _ = self.w_self_trm(w_proj, w_self_attn_mask)

        v2w_cross_attn_mask = get_attn_pad_mask(w_proj, v_proj)
        a2w_cross_attn_mask = get_attn_pad_mask(w_proj, a_proj)
        if self.fixed:
            if self.use_shared_list:
                v2w_cross_hs, all_v2w_cross_attn_weights = self.v2w_cross_tfm(v_proj, w_proj, v2w_cross_attn_mask,
                                                                              all_v_attn_weights, self.shared_list)
                a2w_cross_hs, all_a2w_cross_attn_weights = self.a2w_cross_trm(a_proj, w_proj, a2w_cross_attn_mask,
                                                                              all_a_attn_weights, self.shared_list)
            else:
                v2w_cross_hs, all_v2w_cross_attn_weights = self.v2w_cross_tfm(v_proj, w_proj, v2w_cross_attn_mask, all_v_attn_weights)
                a2w_cross_hs, all_a2w_cross_attn_weights = self.a2w_cross_trm(a_proj, w_proj, a2w_cross_attn_mask, all_a_attn_weights)
        else:
            v2w_cross_hs, all_v2w_cross_attn_weights = self.v2w_cross_tfm(v_proj, w_proj, v2w_cross_attn_mask)
            a2w_cross_hs, all_a2w_cross_attn_weights = self.a2w_cross_trm(a_proj, w_proj, a2w_cross_attn_mask)

        v2w_self_attn_mask = get_attn_pad_mask(v2w_cross_hs, v2w_cross_hs)
        a2w_self_attn_mask = get_attn_pad_mask(a2w_cross_hs, a2w_cross_hs)
        v2w_self_hs, all_v2w_self_attn_weights = self.v2w_self_trm(v2w_cross_hs, v2w_self_attn_mask)
        a2w_self_hs, all_a2w_self_attn_weights = self.a2w_self_trm(a2w_cross_hs, a2w_self_attn_mask)

        concat = torch.cat([v2w_self_hs[:,-1], a2w_self_hs[:,-1]], dim=1)
        out = self.fc(concat)

        return out.view(-1), [all_v2w_cross_attn_weights, all_v2w_self_attn_weights,
                              all_a2w_cross_attn_weights, all_a2w_self_attn_weights]

