#
# code by Zijie Lin @jeff97
# Reference : https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer
#
import numpy as np
import torch
import torch.nn as nn
from modules.transformer import SelfEncoder, Decoder, MaskCrossEncoder, ConcatMaskCrossEncoder


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
        self.enc_proj = nn.Linear(config["source_size"], config["encoder_hidden_size"])
        self.dec_proj = nn.Linear(config["target_size"], config["decoder_hidden_size"])
        self.encoder = SelfEncoder(hidden_size=config["encoder_hidden_size"],
                                   head_size=config["encoder_head_size"],
                                   ff_size=config["ff_size"],
                                   n_heads=config["num_heads"],
                                   n_layers=config["encoder_num_layers"])
        self.decoder = Decoder(hidden_size=config["decoder_hidden_size"],
                               head_size=config["decoder_head_size"],
                               ff_size=config["ff_size"],
                               n_heads=config["num_heads"],
                               n_layers=config["decoder_num_layers"])
        self.projection = nn.Linear(config["decoder_hidden_size"], config["target_size"])

    def __get_attn_parameters(self):
        return [layer.dec_enc_attn.W_q.state_dict() for layer in self.decoder.layers], \
               [layer.dec_enc_attn.W_k.state_dict() for layer in self.decoder.layers]

    def forward(self, source, target):
        enc_inputs = self.enc_proj(source)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_outputs, all_enc_self_attn_weights = self.encoder(enc_inputs, enc_self_attn_mask)       # (b, enc_n, enc_h_d)

        # decode
        dec_inputs = self.dec_proj(target)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_outputs, all_dec_self_attn_weights, all_dec_enc_attn_weights = self.decoder(dec_inputs, enc_outputs,
                                                                                        dec_self_attn_mask,
                                                                                        dec_enc_attn_mask)
        dec_logits = self.projection(dec_outputs)

        # print(all_dec_enc_attn_weights[0].shape)

        return dec_logits, all_dec_enc_attn_weights[0]

    def predict(self, source, target):
        # get decoder inputs
        enc_inputs = self.enc_proj(source)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_outputs, _ = self.encoder(enc_inputs, enc_self_attn_mask)

        pred_target = torch.zeros_like(target)   # (b, n, t_d)
        next_symbol = torch.zeros(pred_target.size(0), pred_target.size(2)).to(pred_target.device)  # (b, t_d)

        for i in range(pred_target.size(1)):
            # print(i)
            pred_target[:,i,:] = next_symbol
            dec_inputs = self.dec_proj(pred_target)  # (b, n, h_d)

            dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
            dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
            dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
            dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

            dec_outputs, _, _ = self.decoder(dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)    # (b, n, h_d)
            projected = self.projection(dec_outputs)    # (b, n, t_d)
            next_symbol = projected[:,i,:].squeeze(1)   # (b, t_d)
        
        # print(pred_target.size())
        
        return self.forward(source, pred_target)
    


class MIMPT(nn.Module):
    def __init__(self, config):
        super(MIMPT, self).__init__()
        self.config = config
        self.w_proj = nn.Linear(config["w_size"], config["hidden_size"])
        self.v_proj = nn.Linear(config["v_size"], config["hidden_size"])
        self.a_proj = nn.Linear(config["a_size"], config["hidden_size"])

        self.vw_mask_cross_trm = MaskCrossEncoder(hidden_size=config["hidden_size"],
                                                  head_size=config["head_size"],
                                                  ff_size=config["ff_size"],
                                                  n_heads=config["num_heads"],
                                                  n_layers=config["num_layers"])
        self.aw_mask_cross_trm = MaskCrossEncoder(hidden_size=config["hidden_size"],
                                                  head_size=config["head_size"],
                                                  ff_size=config["ff_size"],
                                                  n_heads=config["num_heads"],
                                                  n_layers=config["num_layers"])
        # self.self_trm = SelfEncoder(hidden_size=config["hidden_size"]*2,
        #                             head_size=config["head_size"]*2,
        #                             ff_size=config["ff_size"]*2,
        #                             n_heads=config["num_heads"],
        #                             n_layers=config["num_layers"])

        self.fc = nn.Linear(config["hidden_size"]*2, 1)
        # self.fc = nn.Linear(config["hidden_size"]*4, 1)     # for concat

    def forward(self, w, v, a, l, w2v_mi_mask=None, w2a_mi_mask=None, w2v_mp_mask=None, w2a_mp_mask=None):
        w_proj = self.w_proj(w)
        v_proj = self.v_proj(v)
        a_proj = self.a_proj(a)

        vw_cross_attn_mask = get_attn_pad_mask(w_proj, v_proj)
        aw_cross_attn_mask = get_attn_pad_mask(w_proj, a_proj)

        vw_enc_hs, vw_attn_weights_list = self.vw_mask_cross_trm(v_proj, w_proj, vw_cross_attn_mask,
                                                                 w2v_mi_mask, w2v_mp_mask)
        aw_enc_hs, aw_attn_weights_list = self.aw_mask_cross_trm(a_proj, w_proj, aw_cross_attn_mask,
                                                                 w2a_mi_mask, w2a_mp_mask)
        all_v_mi_cross_attn_weights, all_v_mp_cross_attn_weights, all_vw_self_attn_weights = vw_attn_weights_list
        all_a_mi_cross_attn_weights, all_a_mp_cross_attn_weights, all_aw_self_attn_weights = aw_attn_weights_list

        concat = torch.cat([vw_enc_hs[:, -1], aw_enc_hs[:, -1]], dim=-1)
        # concat = torch.cat([vw_enc_hs, aw_enc_hs], dim=-1)

        # self_attn_mask = get_attn_pad_mask(w_proj, w_proj)
        # self_enc_hs, all_self_attn_weights = self.self_trm(concat, self_attn_mask)
        # out = self_enc_hs[:, -1]
        out = self.fc(concat)

        return out.view(-1), [all_v_mi_cross_attn_weights, all_v_mp_cross_attn_weights, all_vw_self_attn_weights,
                              all_a_mi_cross_attn_weights, all_a_mp_cross_attn_weights, all_aw_self_attn_weights]


def get_mi_out(k_v_hs, q_hs, lengths, mi_mask, cross_attn, soft_attn):
    b, k_l, h_d = k_v_hs.size()
    q_l = q_hs.size(1)

    q_reps = q_hs.contiguous().view(b, q_l, 1, h_d).expand(b, q_l, k_l, h_d)
    k_v_reps = k_v_hs.contiguous().view(b, 1, k_l, h_d).expand(b, q_l, k_l, h_d)

    seq_range = torch.arange(0, k_l).long().to(k_v_hs.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(b, k_l)
    seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
    attn_mask = (seq_range_expand < seq_length_expand).unsqueeze(1).expand(b, q_l, k_l)

    _, attn_weights = cross_attn(k_v_reps, q_reps, attn_mask)
    mi_attn_weights = attn_weights * mi_mask

    mi_seq_out = mi_attn_weights.unsqueeze(3).expand_as(k_v_reps) * k_v_reps
    mi_seq_out = torch.sum(mi_seq_out, dim=2)
    mi_out, _ = soft_attn(mi_seq_out)
    return mi_out.unsqueeze(1)


def get_mp_out(hs, mp_mask, soft_attn):
    mp_mask = mp_mask.expand_as(hs)
    sel = hs.masked_select(mp_mask).view(hs.size(0), -1, hs.size(2))
    mp_out, _ = soft_attn(sel)
    return mp_out.unsqueeze(1)

