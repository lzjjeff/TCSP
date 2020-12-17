"""
归一化处理
"""


import os
import yaml
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from read_data import get_dataloader_wy
from preprocessing import make_batch_4_translation_wy, shift_inputs
from model import Translation, MIMPT
from optimization import WarmUpLRScheduler
from util import config
from metrics import mosei_metrics


def information_entropy(attn_weights):  # (b, tgt_n, src_n)
    """ 加入信息熵惩罚项，使tgt对src的attention weight能够区分开 """
    ie = - attn_weights * torch.log2(attn_weights)
    ie = ie.sum(dim=2).mean()
    return ie


def train_translation(tr_loader, val_loader, model, optimizer, scheduler, loss_fun, predix="w2v"):
    """ 训练翻译模型 """
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
            optimizer.zero_grad()
            batch = make_batch_4_translation_wy(batch, predix)
            batch = tuple(t.to(device) for t in batch)
            src, tgt_in, tgt, l = batch
            batch_size = src.size(0)  # （batch_size, length, dim）
            tr_size += batch_size
            tgt_pred, attn_weights = model(src, tgt_in)

            # test = torch.sum(torch.isnan(tgt_pred))
            # if test != 0:
            #     print(f"======================FORWARD ERROR:{test}======================")
            #     exit()

            loss = loss_fun(tgt_pred, tgt) * tgt.size(2)
            loss.backward()
            nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], 0.8)
            optimizer.step()

            # for n, p in model.named_parameters():
            #     test = torch.sum(torch.isnan(p.grad))
            #     if test != 0:
            #         print(f"======================GRAD ERROR:{n}, {test}======================")
            #         exit()

            tr_loss += loss.item() * batch_size
            #print("loss = %f" % loss.data)
        tr_loss = tr_loss / tr_size
        tr_losses.append(tr_loss)
        print("EPOCH %s | Train loss: %s" % (epoch, round(tr_loss, 4)))

        model.eval()
        with torch.no_grad():  # 不进行梯度计算
            val_loss = 0.0
            val_size = 0
            val_attn_weights = []
            for batch in val_loader:
                optimizer.zero_grad()
                batch = make_batch_4_translation_wy(batch, predix)
                batch = tuple(t.to(device) for t in batch)
                src, tgt_in, tgt, l = batch
                batch_size = src.size(0)
                val_size += batch_size
                tgt_pred, attn_weights = model(src, tgt_in)
                loss = loss_fun(tgt_pred, tgt) * tgt.size(2)
                val_loss += loss.item() * batch_size
                val_attn_weights.append(attn_weights)
        val_loss = val_loss / val_size
        val_losses.append(val_loss)
        # val_attn_weights = torch.cat(val_attn_weights, dim=0)
        print("EPOCH %s | Validtation loss: %s\nEPOCH %s | Current learning rate: %s" %
              (epoch, round(val_loss, 4), epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        with open(f'{config["translation"]["result_path"]}translation_result_{predix}.txt', 'a', encoding='utf-8') as fo:
            fo.write("\nEPOCH %s | Train loss: %s\nEPOCH %s | Validtation loss: %s\nEPOCH %s | Current learning rate: %s" %
                     (epoch, round(tr_loss, 4), epoch, round(val_loss, 4), epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            print("A new best model on valid set")
            # 保存attention参数、模型参数和优化器状态
            if device == "cpu":
                torch.save(model.state_dict(), os.path.join(config["translation"]["save_path"], f'translation_model_{predix}.std'))
            else:
                # torch.save(model.module.get_attn_parameters(), f'{config["translation"]["save_path"]}attn_params_{predix}.std')
                torch.save(model.module.state_dict(), os.path.join(config["translation"]["save_path"], f'translation_model_{predix}.std'))
            torch.save(optimizer.state_dict(), os.path.join(config["translation"]["save_path"], f'translation_optim_{predix}.std'))
            print("Current learning rate: %s" % optimizer.state_dict()['param_groups'][0]['lr'])

        scheduler.step(val_loss)
        if optimizer.state_dict()['param_groups'][0]['lr'] < last_lr:
            if device == "cpu":
                model.load_state_dict(torch.load(f'{config["translation"]["save_path"]}translation_model_{predix}.std'))
            else:
                model.module.load_state_dict(torch.load(f'{config["translation"]["save_path"]}translation_model_{predix}.std'))
            last_lr = optimizer.state_dict()['param_groups'][0]['lr']

        lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])

        # 保存val_attn_weights
        torch.save(val_attn_weights, os.path.join(config["translation"]["save_path"], f'attn_weights/val_attn_weights_{predix}_{epoch}.std'))


def get_mi_mask(trans_attn_weights, type='theta'):
    trans_attn_weights = trans_attn_weights.mean(1)     # (b, n_h, v_n, l_n) -> (b, v_n, l_n)
    if type == 'theta':
        mi_mask = trans_attn_weights < config["regression"]["mask_theta"]
        mi_mask = mi_mask.permute(0, 2, 1).contiguous()  # (b, l_n, v_n)
        return mi_mask
    elif type == 'topk':
        _, indices = trans_attn_weights.sort(dim=2, descending=True)
        mi_mask = torch.ones_like(trans_attn_weights)
        mi_mask[:, :, :config["regression"]["mask_topk"]] = 0
        _, indices = indices.sort(dim=-1)
        mi_mask = mi_mask.gather(dim=2, index=indices)
        mi_mask = mi_mask.permute(0, 2, 1).contiguous().to(torch.bool)  # (b, l_n, v_n)
        return mi_mask
    else:
        raise TypeError('Invalid mask shape!')


def get_mp_mask(trans_pred, trans_true, trans_loss_func, type='topk'):
    trans_losses = trans_loss_func(trans_pred, trans_true).mean(-1)     # (b, v_n)
    if type == 'theta':
        mp_mask = trans_losses < config["regression"]["mask_theta"]
        mp_mask = mp_mask.unsqueeze(1).repeat(1, trans_pred.size(1), 1)  # (b, l_n, v_n)
        return mp_mask
    elif type == 'topk':
        _, indices = trans_losses.sort(dim=-1, descending=True)
        mp_mask = torch.ones_like(trans_losses)
        mp_mask[:, :config["regression"]["mask_topk"]] = 0
        _, indices = indices.sort(dim=-1)
        mp_mask = mp_mask.gather(dim=1, index=indices)
        mp_mask = mp_mask.unsqueeze(1).repeat(1, trans_pred.size(1), 1).to(torch.bool)   # (b, l_n, v_n)
        return mp_mask
    else:
        raise TypeError('Invalid mask shape!')


def batch_fit(batch, model, regre_loss_func, trans_model_w2v, trans_model_w2a, trans_loss_func, mode='train'):
    """ 回归模型 batch 处理"""
    batch = tuple(t.to(device) for t in batch)
    w, v, a, v_norm, a_norm, y, l = batch
    y = y.view(-1)
    batch_size = w.size(0)  # （batch_size, length, dim）

    # modality-invariant & modality-private
    v_pred, w2v_attn_weights = trans_model_w2v(w, shift_inputs(v_norm))       # (b, v_n, d), (b, n_h, v_n, l_n)
    a_pred, w2a_attn_weights = trans_model_w2a(w, shift_inputs(a_norm))

    # modality-invariant mask
    w2v_mi_mask = get_mi_mask(w2v_attn_weights, type=config["regression"]["mi_mask_type"])     # (b, l_n, v_n)
    w2a_mi_mask = get_mi_mask(w2a_attn_weights, type=config["regression"]["mi_mask_type"])     # (b, l_n, a_n)

    # modality-private mask
    w2v_mp_mask = get_mp_mask(v_pred, v_norm, trans_loss_func, type=config["regression"]["mp_mask_type"])   # (b, l_n, v_n)
    w2a_mp_mask = get_mp_mask(a_pred, a_norm, trans_loss_func, type=config["regression"]["mp_mask_type"])   # (b, l_n, v_n)

    # y_pred, _ = model(w, v, a, l)   # basemodel
    y_pred, _ = model(w, v, a, l, w2v_mi_mask, w2a_mi_mask, w2v_mp_mask, w2a_mp_mask)
    loss = regre_loss_func(y_pred, y)

    # save attention weights
    # torch.save(w2v_attn_weights, f'{config["regression"]["result_path"]}w2v_attn_weights_b0.std')
    # torch.save(w2a_attn_weights, f'{config["regression"]["result_path"]}w2a_attn_weights_b0.std')

    if mode == 'test':
        return loss, y_pred, y, batch_size

    return loss, y_pred, batch_size


def train_regression(train_loader, valid_loader, model, optimizer, scheduler, regre_loss_func,
                     trans_model_w2v, trans_model_w2a, trans_loss_func):
    """ 训练回归模型 """
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    last_lr = float('inf')
    lrs = []
    for epoch in tqdm(range(config["regression"]["epoch"]), desc="回归模型训练中..."):
        model.train()
        train_loss = 0.0
        train_size = 0

        scheduler.step(epoch)
        print("EPOCH %s | Current learning rate: %s" % (epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        optimizer.zero_grad()

        for batch in tqdm(train_loader):
            loss, _, batch_size = batch_fit(batch, model, regre_loss_func, trans_model_w2v, trans_model_w2a, trans_loss_func)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() * batch_size
            train_size += batch_size
        train_loss = train_loss / train_size
        train_losses.append(train_loss)
        print("EPOCH %s | Train loss: %s" % (epoch, round(train_loss, 4)))

        model.eval()
        with torch.no_grad():  # 不进行梯度计算
            valid_loss = 0.0
            valid_size = 0
            for batch in valid_loader:
                loss, _, batch_size = batch_fit(batch, model, regre_loss_func, trans_model_w2v, trans_model_w2a, trans_loss_func)
                valid_loss += loss.item() * batch_size
                valid_size += batch_size
        valid_loss = valid_loss / valid_size
        valid_losses.append(valid_loss)
        print("EPOCH %s | Validtation loss: %s" % (epoch, round(valid_loss, 4)))

        # 保存结果
        with open(f'{config["regression"]["result_path"]}result.txt',
                  'a', encoding='utf-8') as fo:
            fo.write("\nEPOCH {0} | Train loss: {1}\nEPOCH {0} | Validtation loss: {2}\nEPOCH {0} | Current learning rate: {3}".format(
                epoch, round(train_loss, 4), round(valid_loss, 4), optimizer.state_dict()['param_groups'][0]['lr']))

        # 保存模型
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            print("A new best model on valid set")
            if device == "cpu":
                torch.save(model.state_dict(), f'{config["regression"]["save_path"]}model.std')
            else:
                torch.save(model.module.state_dict(), f'{config["regression"]["save_path"]}model.std')
            torch.save(optimizer.state_dict(), f'{config["regression"]["save_path"]}optim.std')
            # print("Current learning rate: %s" % optimizer.state_dict()['param_groups'][0]['lr'])

        # 更新学习率并读取最佳模型
        # scheduler.step(valid_loss)
        # if optimizer.state_dict()['param_groups'][0]['lr'] < last_lr:
        #     if device == "cpu":
        #         model.load_state_dict(torch.load(f'{config["regression"]["save_path"]}model.std'))
        #     else:
        #         model.module.load_state_dict(torch.load(f'{config["regression"]["save_path"]}model.std'))
        #     last_lr = optimizer.state_dict()['param_groups'][0]['lr']

        lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])

    # 绘制lr曲线
    plt.plot(list(range(config["regression"]["epoch"])), lrs)
    plt.show()


def test_regression(test_loader, model, regre_loss_func, trans_model_w2v, trans_model_w2a, trans_loss_func):
    """ 在测试集上进行预测并输出结果 """

    if device == "cpu":
        model.load_state_dict(torch.load(f'{config["regression"]["save_path"]}model.std'))
    else:
        model.module.load_state_dict(torch.load(f'{config["regression"]["save_path"]}model.std'))

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_size = 0
        for batch in test_loader:
            loss, _y_pred, y, batch_size = batch_fit(batch, model, regre_loss_func, trans_model_w2v, trans_model_w2a,
                                                     trans_loss_func, mode='test')
            y_true.append(y.cpu())
            y_pred.append(_y_pred.cpu())
            test_loss += loss.item() * batch_size
            test_size += batch_size

    print("Test loss: %s" % round(test_loss / test_size, 4))
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    bi_acc, f1, mae, corr = mosei_metrics(y_true, y_pred)
    print("Test set acc_2 is %s\nTest set f1 score is %s\nTest set MAE is %s\nTest set Corr is %s"
          % (bi_acc, f1, mae, corr))

    # 保存结果
    with open(f'{config["regression"]["result_path"]}result.txt',
              'a', encoding='utf-8') as fo:
        fo.write("\nTest loss: %s\nTest set accuracy is %s\nTest set f1 score is %s\n"
                 "Test set MAE is %s\nTest set Corr is %s\n"
                 % (round(test_loss / test_size, 4), bi_acc, f1, mae, corr))


if __name__ == '__main__':
    np.random.seed(config["general"]["seed"])
    torch.manual_seed(config["general"]["seed"])
    torch.cuda.manual_seed_all(config["general"]["seed"])

    device = torch.device("cuda:%s" % config["general"]["device_ids"][0]) if torch.cuda.is_available() else "cpu"

    # 载入数据
    print("载入数据...")
    train_loader, valid_loader, test_loader = get_dataloader_wy(config)

    if config["general"]["do_trans"]:
        # 检查输出目录
        if not os.path.exists(config["translation"]["save_path"]):
            os.mkdir(config["translation"]["save_path"])
        if not os.path.exists(config["translation"]["result_path"]):
            os.mkdir(config["translation"]["result_path"])

        # 创建翻译模型
        print("创建翻译模型...")
        translation_model_w2v = Translation(config["w2v_model"])
        translation_model_w2a = Translation(config["w2a_model"])
        if not device == "cpu":
            translation_model_w2v = nn.DataParallel(translation_model_w2v, device_ids=config["general"]["device_ids"])
            translation_model_w2a = nn.DataParallel(translation_model_w2a, device_ids=config["general"]["device_ids"])
        translation_model_w2v = translation_model_w2v.to(device)
        translation_model_w2a = translation_model_w2a.to(device)

        translation_optimizer_w2v = optim.Adam([param for param in translation_model_w2v.parameters() if param.requires_grad],
                               lr=config["translation"]["lr"], weight_decay=config["translation"]["weight_decay"])
        translation_optimizer_w2a = optim.Adam([param for param in translation_model_w2a.parameters() if param.requires_grad],
                               lr=config["translation"]["lr"], weight_decay=config["translation"]["weight_decay"])

        translation_loss_func = nn.MSELoss()
        scheduler_w2v = ReduceLROnPlateau(translation_optimizer_w2v, mode='min', patience=5, factor=0.1, verbose=True)
        scheduler_w2a = ReduceLROnPlateau(translation_optimizer_w2a, mode='min', patience=5, factor=0.1, verbose=True)

        # 读取checkpoint文件
        # if config["translation"]["from_trained"]:
        #     if device == "cpu":
        #         translation_model_w2v.load_state_dict(
        #             torch.load(f'{config["translation"]["save_path"]}translation_model_w2v.std'))
        #         translation_model_w2a.load_state_dict(
        #             torch.load(f'{config["translation"]["save_path"]}translation_model_w2a.std'))
        #     else:
        #         translation_model_w2v.module.load_state_dict(
        #             torch.load(f'{config["translation"]["save_path"]}translation_model_w2v.std'))
        #         translation_model_w2a.module.load_state_dict(
        #             torch.load(f'{config["translation"]["save_path"]}translation_model_w2a.std'))
        #     translation_optimizer_w2v.load_state_dict(
        #         torch.load(f'{config["translation"]["save_path"]}translation_optim_w2v.std'))
        #     translation_optimizer_w2a.load_state_dict(
        #         torch.load(f'{config["translation"]["save_path"]}translation_optim_w2a.std'))
        #
        #     # 更新路径
        #     config["translation"]["save_path"] += "01/"
        #     config["translation"]["result_path"] += "01/"
        #     if not os.path.exists(config["translation"]["save_path"]):
        #         os.mkdir(config["translation"]["save_path"])
        #     if not os.path.exists(config["translation"]["result_path"]):
        #         os.mkdir(config["translation"]["result_path"])

        # 保存参数文件
        with open('{}/config.yaml'.format(config["translation"]["result_path"]), 'w') as f:
            yaml.dump(config, f)

        # 训练翻译模型
        print("训练翻译模型...")
        #train_translation(train_loader, valid_loader, translation_model_w2v, translation_optimizer_w2v, scheduler_w2v,
        #                  translation_loss_func, predix='w2v')
        train_translation(train_loader, valid_loader, translation_model_w2a, translation_optimizer_w2a, scheduler_w2a,
                          translation_loss_func, predix='w2a')

    if config["general"]["do_regre"]:
        # 检查输出目录
        if not os.path.exists(config["regression"]["save_path"]):
            os.mkdir(config["regression"]["save_path"])
        if not os.path.exists(config["regression"]["result_path"]):
            os.mkdir(config["regression"]["result_path"])

        # 保存参数文件
        with open('{}/config.yaml'.format(config["regression"]["result_path"]), 'w') as f:
            yaml.dump(config, f)

        # 创建回归模型
        print("创建回归模型...")
        regression_model = MIMPT(config["regression"])
        if not device == "cpu":
            regression_model = nn.DataParallel(regression_model, device_ids=config["general"]["device_ids"])
        regression_model = regression_model.to(device)

        regression_optimizer = optim.Adam([param for param in regression_model.parameters() if param.requires_grad],
                               lr=config["regression"]["lr"], weight_decay=config["regression"]["weight_decay"])
        # scheduler = ReduceLROnPlateau(regression_optimizer, mode='min', patience=5, factor=0.1, verbose=True)
        scheduler = WarmUpLRScheduler(regression_optimizer, warmup=0.1, total_epochs=config["regression"]["epoch"])
        regresson_loss_func = nn.L1Loss()

        trans_loss_func = nn.MSELoss(reduce=False)

        # 创建翻译模型
        translation_model_w2v = Translation(config["w2v_model"])
        translation_model_w2a = Translation(config["w2a_model"])
        if not device == "cpu":
            translation_model_w2v = nn.DataParallel(translation_model_w2v, device_ids=config["general"]["device_ids"])
            translation_model_w2a = nn.DataParallel(translation_model_w2a, device_ids=config["general"]["device_ids"])
        translation_model_w2v = translation_model_w2v.to(device)
        translation_model_w2a = translation_model_w2a.to(device)

        # 读取state_dict文件
        # regression_model.module.load_state_dict(torch.load(f'{config["regression"]["save_path"]}model.std'))
        # regression_optimizer.load_state_dict(torch.load(f'{config["regression"]["save_path"]}optim.std'))

        if device == "cpu":
            translation_model_w2v.load_state_dict(
                torch.load(f'{config["translation"]["save_path"]}translation_model_w2v.std'))
            translation_model_w2a.load_state_dict(
                torch.load(f'{config["translation"]["save_path"]}translation_model_w2a.std'))
        else:
            translation_model_w2v.module.load_state_dict(
                torch.load(f'{config["translation"]["save_path"]}translation_model_w2v.std'))
            translation_model_w2a.module.load_state_dict(
                torch.load(f'{config["translation"]["save_path"]}translation_model_w2a.std'))

        # 固定translation模型参数
        for n, p in translation_model_w2v.named_parameters():
            print(n)
            p.requires_grad = False
        for n, p in translation_model_w2a.named_parameters():
            print(n)
            p.requires_grad = False

        # 训练回归模型
        print("训练回归模型...")
        train_regression(train_loader, valid_loader, regression_model, regression_optimizer, scheduler,
                        regresson_loss_func, translation_model_w2v, translation_model_w2a, trans_loss_func)

        regression_model = MIMPT(config["regression"])
        if not device == "cpu":
            regression_model = nn.DataParallel(regression_model, device_ids=config["general"]["device_ids"])
        regression_model = regression_model.to(device)

        test_regression(test_loader, regression_model, regresson_loss_func,
                        translation_model_w2v, translation_model_w2a, trans_loss_func)

