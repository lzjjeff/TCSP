import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import yaml
import csv
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from transformer.read_data import get_dataloader
from transformer.preprocessing import multi_collate, make_batch_4_translation, shift_inputs
from transformer.model import Translation, DualAttentionRegressoin
from transformer.util import config


def train_translation(tr_loader, val_loader, model, optimizer, scheduler, loss_fun, predix="v2w"):
    tr_losses = []
    val_losses = []
    best_val_loss = float('inf')
    last_lr = float('inf')
    for epoch in tqdm(range(config["translation"]["epoch"]), desc="翻译模型训练中..."):
        lrs = []
        model.train()
        tr_loss = 0.0
        tr_size = 0
        for batch in tqdm(tr_loader):
            model.zero_grad()
            batch = make_batch_4_translation(batch, predix)
            batch = [tuple(t.to(device) for t in batch[0]), batch[1]]
            (src, tgt_in, tgt), meta = batch
            batch_size = src.size(0)  # （batch_size, length, dim）
            tr_size += batch_size
            tgt_pred, _ = model(src, tgt_in)
            loss = loss_fun(tgt_pred, tgt) * tgt.size(2)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * batch_size
            # print("loss = %f" % loss.data)
        tr_loss = tr_loss / tr_size
        tr_losses.append(tr_loss)
        print("EPOCH %s | Train loss: %s" % (epoch, round(tr_loss, 4)))

        model.eval()
        with torch.no_grad():  # 不进行梯度计算
            val_loss = 0.0
            val_size = 0
            for batch in val_loader:
                model.zero_grad()
                batch = make_batch_4_translation(batch, predix)
                batch = [tuple(t.to(device) for t in batch[0]), batch[1]]
                (src, tgt_in, tgt), meta = batch
                batch_size = src.size(0)
                val_size += batch_size
                tgt_pred, _ = model(src, tgt_in)
                loss = loss_fun(tgt_pred, tgt) * tgt.size(2)
                val_loss += loss.item() * batch_size
        val_loss = val_loss / val_size
        val_losses.append(val_loss)
        print("EPOCH %s | Validtation loss: %s\nEPOCH %s | Current learning rate: %s" %
              (epoch, round(val_loss, 4), epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        with open(f'{config["general"]["result_path"]}translation_result_{predix}.txt', 'a', encoding='utf-8') as fo:
            fo.write("\nEPOCH %s | Train loss: %s\nEPOCH %s | Validtation loss: %s\nEPOCH %s | Current learning rate: %s" %
                     (epoch, round(tr_loss, 4), epoch, round(val_loss, 4), epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            print("A new best model on valid set")
            # 保存模型参数和优化器状态
            torch.save(model.module.state_dict(), f'{config["general"]["save_path"]}translation_model_{predix}.std')
            torch.save(optimizer.state_dict(), f'{config["general"]["save_path"]}translation_optim_{predix}.std')
            print("Current learning rate: %s" % optimizer.state_dict()['param_groups'][0]['lr'])

        scheduler.step(val_loss)
        if optimizer.state_dict()['param_groups'][0]['lr'] < last_lr:
            model.module.load_state_dict(torch.load(f'{config["general"]["save_path"]}translation_model_{predix}.std'))
            last_lr = optimizer.state_dict()['param_groups'][0]['lr']

        lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])


def train_regression(train_loader, valid_loader, test_loader, reg_model, optimizer, scheduler, loss_func,
                     trans_model_v2w=None, trans_model_a2w=None):
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    last_lr = float('inf')
    for epoch in tqdm(range(config["regression"]["epoch"]), desc="回归模型训练中..."):
        lrs = []
        reg_model.train()
        train_loss = 0.0
        train_size = 0
        for batch in tqdm(train_loader):
            reg_model.zero_grad()
            batch = [tuple(t.to(device) for t in batch[0]), batch[1].to(device), batch[2]]
            (i, w, a, v), y, meta = batch
            y = y.view(-1)
            # w, v, a = tuple(t.permute(1, 0, 2).contiguous() for t in (w, v, a))  # (b, n ,d)
            batch_size = w.size(0)  # （batch_size, length, dim）
            train_size += batch_size
            if config["regression"]["fixed"]:
                w_tgt = shift_inputs(w).to(device)
                _, all_attn_weights_v2w = trans_model_v2w(v, w_tgt)
                _, all_attn_weights_a2w = trans_model_a2w(a, w_tgt)
                y_pred, _ = reg_model(w, v, a, all_attn_weights_v2w, all_attn_weights_a2w)
            else:
                y_pred, _ = reg_model(w, v, a)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size
        train_loss = train_loss / train_size
        train_losses.append(train_loss)
        print("EPOCH %s | Train loss: %s" % (epoch, round(train_loss, 4)))

        reg_model.eval()
        with torch.no_grad():  # 不进行梯度计算
            valid_loss = 0.0
            valid_size = 0
            for batch in valid_loader:
                reg_model.zero_grad()
                batch = [tuple(t.to(device) for t in batch[0]), batch[1].to(device), batch[2]]
                (i, w, a, v), y, meta = batch
                y = y.view(-1)
                # w, v, a = tuple(t.permute(1, 0, 2).contiguous() for t in (w, v, a))
                batch_size = w.size(0)
                valid_size += batch_size
                if config["regression"]["fixed"]:
                    w_tgt = shift_inputs(w).to(device)
                    _, all_attn_weights_v2w = trans_model_v2w(v, w_tgt)
                    _, all_attn_weights_a2w = trans_model_a2w(a, w_tgt)
                    y_pred, _ = reg_model(w, v, a, all_attn_weights_v2w, all_attn_weights_a2w)
                else:
                    y_pred, _ = reg_model(w, v, a)
                loss = loss_func(y_pred, y)
                valid_loss += loss.item() * batch_size
        valid_loss = valid_loss / valid_size
        valid_losses.append(valid_loss)
        print("EPOCH %s | Validtation loss: %s" % (epoch, round(valid_loss, 4)))

        if config["regression"]["fixed"]:
            test_regression(test_loader, reg_model, loss_func, trans_model_v2w,
                            trans_model_a2w, training=True)
        else:
            test_regression(test_loader, reg_model, loss_func, training=True)

        # 写入文件
        if config["regression"]["model_type"] == "attention":
            suffix = "fix" if config["regression"]["fixed"] else "fle"
        else:
            suffix = "fix_fle" if config["regression"]["fixed"] else "fle_fle"
        with open(f'{config["general"]["result_path"]}result_{suffix}.txt', 'a', encoding='utf-8') as fo:
            fo.write("\nEPOCH {0} | Train loss: {1}\nEPOCH {0} | Validtation loss: {2}\nEPOCH {0} | Current learning rate: {3}".format(
                epoch, round(train_loss, 4), round(valid_loss, 4), optimizer.state_dict()['param_groups'][0]['lr']))

        # 判断最优模型是否更新
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            print("A new best model on valid set")
            torch.save(reg_model.module.state_dict(), f'{config["general"]["result_path"]}model.std')
            torch.save(optimizer.state_dict(), f'{config["general"]["result_path"]}optim.std')
            print("Current learning rate: %s" % optimizer.state_dict()['param_groups'][0]['lr'])

        # 判断是否更新lr
        scheduler.step(valid_loss)
        if optimizer.state_dict()['param_groups'][0]['lr'] < last_lr:
            reg_model.module.load_state_dict(torch.load(f'{config["general"]["result_path"]}model.std'))
            last_lr = optimizer.state_dict()['param_groups'][0]['lr']

        lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])


def test_regression(test_loader, reg_model, loss_func, trans_model_v2w=None, trans_model_a2w=None, training=False):
    def metrics(y_true, y_pred):
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])

        y_true_bin = y_true[non_zeros] > 0
        y_pred_bin = y_pred[non_zeros] > 0

        bi_acc = accuracy_score(y_true_bin, y_pred_bin)
        f1 = f1_score(y_true_bin, y_pred_bin, average='weighted')
#        multi_acc = np.round(sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true)), 7)[0]
        mae = mean_absolute_error(y_true, y_pred)
        corr = np.corrcoef(y_pred.reshape(-1), y_true.reshape(-1))[0][1]
        return bi_acc, f1, mae, corr

    ids = []
    metas = [[], [], []]
    y_true = []
    y_pred = []
    reg_model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_size = 0
        for batch in test_loader:
            reg_model.zero_grad()
            batch = [tuple(t.to(device) for t in batch[0]), batch[1].to(device), batch[2]]
            (i, w, a, v), y, meta = batch
            y = y.view(-1)
            # w, v, a = tuple(t.permute(1, 0, 2).contiguous() for t in (w, v, a))
            batch_size = w.size(0)
            test_size += batch_size
            if config["regression"]["fixed"]:
                w_tgt = shift_inputs(w).to(device)
                _, all_attn_weights_v2w = trans_model_v2w(v, w_tgt)
                _, all_attn_weights_a2w = trans_model_a2w(a, w_tgt)
                _y_pred, _ = reg_model(w, v, a, all_attn_weights_v2w, all_attn_weights_a2w)
            else:
                _y_pred, _ = reg_model(w, v, a)
            loss = loss_func(_y_pred, y)
            test_loss += loss.item() * batch_size

            y_true.append(y.cpu())
            y_pred.append(_y_pred.cpu())
            ids.append(i.view(-1).cpu())
            metas[0].extend(meta[0])
            metas[1].extend(meta[1])
            metas[2].extend(meta[2])

    print("Test loss: %s" % round(test_loss / test_size, 4))

    if not training:
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        bi_acc, f1, mae, corr = metrics(y_true, y_pred)
        print("Test set acc_2 is %s\nTest set f1 score is %s\nTest set MAE is %s\nTest set Corr is %s"
              % (bi_acc, f1, mae, corr))

        # 写入文件
        if config["regression"]["model_type"] == "attention":
            suffix = "fix" if config["regression"]["fixed"] else "fle"
        else:
            suffix = "fix_fle" if config["regression"]["fixed"] else "fle_fle"
        with open(f'{config["general"]["result_path"]}result_{suffix}.txt', 'a', encoding='utf-8') as fo:
            fo.write("\nTest loss: %s\nTest set accuracy is %s\nTest set f1 score is %s\n"
                     "Test set MAE is %s\nTest set Corr is %s\n"
                     % (round(test_loss / test_size, 4), bi_acc, f1, mae, corr))

        ids = torch.cat(ids, dim=0).numpy()
        segments = np.array(metas[0])
        starts = np.array(metas[1])
        ends = np.array(metas[2])

        output_pd = pd.DataFrame({"id": ids,
                                  "true sen": y_true,
                                  "pred sen": y_pred,
                                  "segment": segments,
                                  "start": starts,
                                  "end": ends})

        sorted_pd = output_pd.sort_values(by=["id"])
        output_path = os.path.join(config["general"]["result_path"], f"test_outputs_{suffix}.csv")
        sorted_pd.to_csv(output_path)


if __name__ == '__main__':
    np.random.seed(config["general"]["seed"])
    torch.manual_seed(config["general"]["seed"])
    torch.cuda.manual_seed_all(config["general"]["seed"])

    device = torch.device("cuda:%s" % config["general"]["device_ids"][0]) if torch.cuda.is_available() else "cpu"

    # 检查输出目录
    if not os.path.exists(config["general"]["save_path"]):
        os.mkdir(config["general"]["save_path"])
    if not os.path.exists(config["general"]["result_path"]):
        os.mkdir(config["general"]["result_path"])

    # 保存参数文件
    with open('{}/config.yaml'.format(config["general"]["result_path"]), 'w') as f:
        yaml.dump(config, f)

    # 载入数据
    print("载入数据...")
    train_loader, valid_loader, test_loader = get_dataloader(config)
    # train, valid, test = pickle.load(open('../MMSA_demo/data/{0}/{0}_emb.pkl'.format(config["general"]["dataset"]), 'rb'))
    # train_loader = Data.DataLoader(train, batch_size=16, shuffle=True, collate_fn=lambda batch: multi_collate(batch, config["translation"]["max_len"]))
    # valid_loader = Data.DataLoader(valid, batch_size=16, shuffle=True, collate_fn=lambda batch: multi_collate(batch, config["translation"]["max_len"]))
    # test_loader = Data.DataLoader(test, batch_size=16, shuffle=True, collate_fn=lambda batch: multi_collate(batch, config["translation"]["max_len"]))

    if config["general"]["do_trans"]:
        # 创建翻译模型
        print("创建翻译模型...")
        translation_model_v2w = Translation(config["v2w_model"])
        translation_model_v2w = nn.DataParallel(translation_model_v2w, device_ids=config["general"]["device_ids"])
        translation_model_v2w.to(device)
        translation_optimizer_v2w = optim.Adam([param for param in translation_model_v2w.parameters() if param.requires_grad],
                               lr=config["translation"]["lr"], weight_decay=config["translation"]["weight_decay"])

        translation_model_a2w = Translation(config["a2w_model"])
        translation_model_a2w = nn.DataParallel(translation_model_a2w, device_ids=config["general"]["device_ids"])
        translation_model_a2w.to(device)
        translation_optimizer_a2w = optim.Adam([param for param in translation_model_a2w.parameters() if param.requires_grad],
                               lr=config["translation"]["lr"], weight_decay=config["translation"]["weight_decay"])
        translation_loss_func = nn.MSELoss()
        scheduler_v2w = ReduceLROnPlateau(translation_optimizer_v2w, mode='min', patience=20, factor=0.1, verbose=True)
        scheduler_a2w = ReduceLROnPlateau(translation_optimizer_a2w, mode='min', patience=20, factor=0.1, verbose=True)

        # # 读取checkpoint文件
        # translation_model_v2w.module.load_state_dict(
        #     torch.load(f'{config["general"]["save_path"]}{config["general"]["subpath"]}translation_model_v2w.std'))
        # translation_model_a2w.module.load_state_dict(
        #     torch.load(f'{config["general"]["save_path"]}{config["general"]["subpath"]}translation_model_a2w.std'))
        # translation_optimizer_v2w.load_state_dict(
        #     torch.load(f'{config["general"]["save_path"]}{config["general"]["subpath"]}translation_optim_v2w.std'))
        # translation_optimizer_a2w.load_state_dict(
        #     torch.load(f'{config["general"]["save_path"]}{config["general"]["subpath"]}translation_optim_a2w.std'))

        # 训练翻译模型
        print("训练翻译模型...")
        train_translation(train_loader, valid_loader, translation_model_v2w, translation_optimizer_v2w, scheduler_v2w,
                          translation_loss_func, predix='v2w')
        train_translation(train_loader, valid_loader, translation_model_a2w, translation_optimizer_a2w, scheduler_a2w,
                          translation_loss_func, predix='a2w')

    if config["general"]["do_regre"]:
        # 创建回归模型
        print("创建回归模型...")
        if config["regression"]["model_type"] == "attention":
            regression_model = AttentionRegressoin(config["regression"])
        elif config["regression"]["model_type"] == "dual_attention":
            regression_model = DualAttentionRegressoin(config["regression"])
        else:
            raise ValueError("Invalid model_type!")
        regression_model = nn.DataParallel(regression_model, device_ids=config["general"]["device_ids"])
        regression_optimizer = optim.Adam([param for param in regression_model.parameters() if param.requires_grad],
                               lr=config["regression"]["lr"], weight_decay=config["regression"]["weight_decay"])
        scheduler = ReduceLROnPlateau(regression_optimizer, mode='min', patience=20, factor=0.1, verbose=True)
        regresson_loss_func = nn.L1Loss()

        if config["regression"]["fixed"]:
            # 创建翻译模型
            translation_model_v2w = Translation(config["v2w_model"])
            translation_model_a2w = Translation(config["a2w_model"])

            translation_model_v2w = nn.DataParallel(translation_model_v2w, device_ids=config["general"]["device_ids"])
            translation_model_v2w.to(device)
            translation_model_a2w = nn.DataParallel(translation_model_a2w, device_ids=config["general"]["device_ids"])
            translation_model_a2w.to(device)

            # 读取state_dict文件
            translation_model_v2w.module.load_state_dict(
                torch.load(f'{config["general"]["save_path"]}translation_model_v2w.std'))
            translation_model_a2w.module.load_state_dict(
                torch.load(f'{config["general"]["save_path"]}translation_model_a2w.std'))

            for n, p in translation_model_a2w.named_parameters():
                print(n)
                p.requires_grad = False

            # 训练回归模型
            print("训练回归模型...")
            train_regression(train_loader, valid_loader, test_loader, regression_model, regression_optimizer, scheduler,
                            regresson_loss_func, translation_model_v2w, translation_model_a2w)

            regression_model.module.load_state_dict(torch.load(f'{config["general"]["result_path"]}model.std'))
            test_regression(test_loader, regression_model, regresson_loss_func, translation_model_v2w,
                            translation_model_a2w)

        else:
            # 训练回归模型
            print("训练回归模型...")
            train_regression(train_loader, valid_loader, test_loader, regression_model, regression_optimizer, scheduler,
                             regresson_loss_func)

            regression_model.module.load_state_dict(torch.load(f'{config["general"]["result_path"]}model.std'))
            test_regression(test_loader, regression_model, regresson_loss_func)


