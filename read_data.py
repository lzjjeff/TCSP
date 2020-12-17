import yaml
import pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data as Data
from MulT.src.dataset import Multimodal_Datasets


def get_dataloader_mult(config):
    """ 读取 MulT 的数据 """

    data_path = "/users6/zjlin/tools/my_pkgs/MulT/data/"
    if config["general"]["dataset"] == "mosi":
        data_name = "mosi"
    elif config["general"]["dataset"] == "mosei":
        data_name = "mosei_senti"
    else:
        raise ValueError("data_path not found")

    train = Multimodal_Datasets(data_path, data_name, "train", True)
    valid = Multimodal_Datasets(data_path, data_name, "valid", True)
    test = Multimodal_Datasets(data_path, data_name, "test", True)

    train_loader = Data.DataLoader(train, batch_size=config["general"]["batch_size"], shuffle=True)
    valid_loader = Data.DataLoader(valid, batch_size=config["general"]["batch_size"], shuffle=True)
    test_loader = Data.DataLoader(test, batch_size=config["general"]["batch_size"], shuffle=True)

    return train_loader, valid_loader, test_loader


from dataset import MoseiDataset, collate_fn
from dataset_splits import mosei_folds

def get_dataloader_wy(config):
    """ 读取 吴洋 的数据 """

    with open('./data/MOSEI/mosei.dataset', 'rb') as f:
        data_dic = pickle.load(f)

    # normalize
    train_fold = mosei_folds['train'] + mosei_folds['valid']
    train_keys = [key for key in data_dic.keys() if key.split("[")[0] in train_fold]
    v_max = np.max(np.array([np.max(np.abs(data_dic[key]['v']), axis=0) for key in train_keys]), axis=0)
    a_max = np.max(np.array([np.max(np.abs(data_dic[key]['a']), axis=0) for key in train_keys]), axis=0)
    v_max[v_max == 0] = 1
    a_max[a_max == 0] = 1

    train_dataset = MoseiDataset(data_dic, 'train', v_max, a_max)
    valid_dataset = MoseiDataset(data_dic, 'valid', v_max, a_max)
    test_dataset = MoseiDataset(data_dic, 'test', v_max, a_max)

    train_loader = Data.DataLoader(train_dataset, batch_size=config["general"]["batch_size"], collate_fn=collate_fn,
                                   shuffle=True)
    valid_loader = Data.DataLoader(valid_dataset, batch_size=config["general"]["batch_size"], collate_fn=collate_fn,
                                   shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=config["general"]["batch_size"], collate_fn=collate_fn,
                                   shuffle=True)

    return train_loader, valid_loader, test_loader