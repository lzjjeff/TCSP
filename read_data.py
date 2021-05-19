import yaml
import pickle
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data as Data
from dataset import MoseiDataset, collate_fn
from dataset_splits import mosei_folds, mosi_folds

def get_dataloader(config):

    if config["general"]["dataset"] == "mosi":
        with open('./data/MOSI/mosi.dataset', 'rb') as f:
            data_dic = pickle.load(f)
        data_folds = mosi_folds
        train_fold = data_folds['train'] + data_folds['valid']
        train_keys = [key for key in data_dic.keys() if "_".join(key.split("_")[:-1]) in train_fold]
    else:
        with open('./data/MOSEI/mosei.dataset', 'rb') as f:
            data_dic = pickle.load(f)
        data_folds = mosei_folds
        train_fold = data_folds['train'] + data_folds['valid']
        train_keys = [key for key in data_dic.keys() if key.split('[')[0] in train_fold]

    # normalize
    v_max = np.max(np.array([np.max(np.abs(data_dic[key]['v']), axis=0) for key in train_keys]), axis=0)
    a_max = np.max(np.array([np.max(np.abs(data_dic[key]['a']), axis=0) for key in train_keys]), axis=0)
    v_max[v_max == 0] = 1
    a_max[a_max == 0] = 1

    train_dataset = MoseiDataset(data_dic, config["general"]["dataset"], 'train', v_max, a_max)
    valid_dataset = MoseiDataset(data_dic, config["general"]["dataset"], 'valid', v_max, a_max)
    test_dataset = MoseiDataset(data_dic, config["general"]["dataset"], 'test', v_max, a_max)

    print("\tNumber of train samples: %d\n\tNumber of valid samples: %d\n\tNumber of test samples: %d"
          % (len(train_dataset), len(valid_dataset), len(test_dataset)))

    train_loader = Data.DataLoader(train_dataset, batch_size=config["general"]["batch_size"], collate_fn=collate_fn,
                                   shuffle=True)
    valid_loader = Data.DataLoader(valid_dataset, batch_size=config["general"]["batch_size"], collate_fn=collate_fn,
                                   shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=config["general"]["batch_size"], collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader
