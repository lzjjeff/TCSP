
import torch


def shift_inputs(inputs):
    """ 将输入序列向右偏移一位，第一位补0
        For transformer
    """
    start = torch.zeros(inputs.size(0), 1, inputs.size(2)).to(inputs.device)
    new_inputs = torch.cat([start, inputs], dim=1)[:, :-1, :]

    return new_inputs


def make_batch_4_translation_mult(batch, predix="w2v"):
    """ 对 MulT 的数据进行整理用于翻译任务 """
    (i, w, a, v), y, meta = batch
    l = torch.LongTensor([w.size(1)]*w.size(0))

    if predix == "w2v":
        v_in = shift_inputs(v)  # (b, n, d)
        return (w, v_in, v, l), meta     # v: (b, n, d)
    if predix == "w2a":
        a_in = shift_inputs(a)  # (b, n, d)
        return (w, a_in, a, l), meta     # a: (b, n, d)


def make_batch_4_translation_wy(batch, predix="w2v"):
    """ 对 吴洋 的数据进行整理用于翻译任务 """
    w, v, a, v_norm, a_norm, y, l = batch

    if predix == "w2v":
        v_in = shift_inputs(v_norm)  # (b, n, d)
        return w, v_in, v_norm, l     # v: (b, n, d)
    if predix == "w2a":
        a_in = shift_inputs(a_norm)  # (b, n, d)
        return w, a_in, a_norm, l    # a: (b, n, d)
