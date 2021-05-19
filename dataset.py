from torch.utils.data import Dataset
from dataset_splits import mosei_folds, mosi_folds
import torch
import pickle
import numpy as np

class MoseiDataset(Dataset):
    def __init__(self, data_dic, dataset="mosei", dname="train", v_max=1.0, a_max=1.0, n_samples=None):
        self.data_dic = data_dic
        if dataset == 'mosi':
            fold = mosi_folds[dname]
            self.fold_keys = [key for key in data_dic.keys() if "_".join(key.split("_")[:-1]) in fold]
        else:
            fold = mosei_folds[dname]
            self.fold_keys = [key for key in data_dic.keys() if key.split('[')[0] in fold]
        if n_samples:
            self.fold_keys = self.fold_keys[:n_samples]
        self.v_max = v_max
        self.a_max = a_max

    def __getitem__(self, idx):
        key = self.fold_keys[idx]

        audio = self.data_dic[key]['a']
        audio[~np.isfinite(audio)] = 0
        audio_normed = audio / self.a_max

        video = self.data_dic[key]['v']
        video[~np.isfinite(video)] = 0
        video_normed = video / self.v_max

        data = {"id": key, "text": self.data_dic[key]['t'], "video": video,
                "audio": audio, "video_normed": video_normed,
                "audio_normed": audio_normed, "label": self.data_dic[key]['l'][0]}
        return data

    def __len__(self):
        return len(self.fold_keys)


def sort_sequences(inputs, lengths):
    """sort_sequences
    Sort sequences according to lengths descendingly.

    :param inputs (Tensor): input sequences, size [B, T, D]
    :param lengths (Tensor): length of each sequence, size [B]
    """
    lengths = torch.Tensor(lengths)
    lengths_sorted, sorted_idx = lengths.sort(descending=True)
    _, unsorted_idx = sorted_idx.sort()
    return inputs[sorted_idx], lengths_sorted, sorted_idx


def collate_fn(batch):
    MAX_LEN = 128

    lens = [min(len(row["text"]), MAX_LEN) for row in batch]

    tdims = batch[0]["text"].shape[1]
    adims = batch[0]["audio"].shape[1]
    vdims = batch[0]["video"].shape[1]

    bsz, max_seq_len = len(batch), max(lens)

    text_tensor = torch.zeros((bsz, max_seq_len, tdims))

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = torch.Tensor(input_row["text"][:length])

    video_tensor = torch.zeros((bsz, max_seq_len, vdims))
    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        video_tensor[i_batch, :length] = torch.Tensor(input_row["video"][:length])


    audio_tensor = torch.zeros((bsz, max_seq_len, adims))
    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        audio_tensor[i_batch, :length] = torch.Tensor(input_row["audio"][:length])

    video_normed_tensor = torch.zeros((bsz, max_seq_len, vdims))
    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        video_normed_tensor[i_batch, :length] = torch.Tensor(input_row["video_normed"][:length])

    audio_normed_tensor = torch.zeros((bsz, max_seq_len, adims))
    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        audio_normed_tensor[i_batch, :length] = torch.Tensor(input_row["audio_normed"][:length])

    tgt_tensor = torch.stack([torch.tensor(row["label"]) for row in batch])

    text_tensor, lens, sorted_idx = sort_sequences(text_tensor, lens)

    return text_tensor, video_tensor[sorted_idx], audio_tensor[sorted_idx], \
           video_normed_tensor[sorted_idx], audio_normed_tensor[sorted_idx], tgt_tensor[sorted_idx], lens


if __name__ == "__main__":

    with open("data/MOSEI/mosei.dataset", "rb") as f:
        data_dic = pickle.load(f)

    train_dataset = MoseiDataset(data_dic, "train")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

    for text_tensor, video_tensor, audio_tensor, tgt_tensor, video_normed_tensor, audio_normed_tensor, lens in train_loader:
        print(text_tensor.size(), video_tensor.size(), audio_tensor.size(),
              video_normed_tensor.size(), audio_normed_tensor.size(), tgt_tensor.size(), lens.size())

        exit()




