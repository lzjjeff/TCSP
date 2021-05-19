
import torch


def shift_inputs(inputs):
    """ Shift the input sequence to the right by one position, the first position is complemented by 0 """
    start = torch.zeros(inputs.size(0), 1, inputs.size(2)).to(inputs.device)
    new_inputs = torch.cat([start, inputs], dim=1)[:, :-1, :]

    return new_inputs


def make_batch_4_translation(config, batch, predix='w2v'):
    w, v, a, v_n, a_n, y, l = batch
    w_in = shift_inputs(w)  # (b, n, d)

    if predix == "w2v":
        return w, v_n, l  # v: (b, n, d)
    if predix == "w2a":
        return w, a_n, l  # a: (b, n, d)


def make_batch_4_regression(config, batch):
    return batch