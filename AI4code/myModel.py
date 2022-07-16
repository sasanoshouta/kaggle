import os
import gc
import sys
import json
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy import sparse
from bisect import bisect
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from config import CFG
from functions import *

# ====================================================
# Models
# ====================================================


# class MarkdownModel(nn.Module):
#     def __init__(self, model_path):
#         super(MarkdownModel, self).__init__()
#         self.model = AutoModel.from_pretrained(model_path)
#         self.top = nn.Linear(769, 1)

#     def forward(self, ids, mask, fts):
#         x = self.model(ids, mask)[0]
#         x = torch.cat((x[:, 0, :], fts), 1)
#         x = self.top(x)
#         return x

class MarkdownModel(nn.Module):
    def __init__(self, cfg, trained_model_path=None):
        super(MarkdownModel, self).__init__()
        self.cfg = cfg
        if trained_model_path is None:
            self.config = AutoConfig.from_pretrained(cfg.BERT_PATH, output_hidden_states=True)
        else:
            self.config = torch.load(trained_model_path+"config.pth")
        if trained_model_path is None:
            self.distill_bert = AutoModel.from_pretrained(self.cfg.BERT_PATH)
        else:
            self.distill_bert = AutoModel.from_config(self.config)
        self.top = nn.Linear(769, 1)
        # self.top = nn.Linear(1025, 1)
        
    def forward(self, ids, mask, fts):
        x = self.distill_bert(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts), 1)
        # x = self.top(x[:, 0, :])
        x = self.top(x)
        return x

def adjust_lr(optimizer, epoch):
    if epoch < 1:
        lr = 5e-5
    elif epoch < 2:
        lr = 1e-3
    elif epoch < 5:
        lr = 1e-4
    else:
        lr = 1e-5

    for p in optimizer.param_groups:
        p['lr'] = lr
    return lr
    
def get_optimizer(net):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-4, betas=(0.9, 0.999),
                                 eps=1e-08)
    return optimizer

def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler

def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(LOGGER, model, val_loader, epoch):
    model.eval()
    # tbar = tqdm(val_loader, file=sys.stdout)
    start = end = time.time()
    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            inputs, target = read_data(data)

            # pred = model(inputs[0], inputs[1])
            pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            end = time.time()
            remain = timeSince(start, float(idx+1)/len(val_loader))
            if idx % CFG.print_freq == 0 or idx == (len(val_loader)-1):
                # LOGGER.info(f"Epoch {e+1} Steps: [{idx}/{len(train_loader)}] Elapse: {remain:s} Loss: {avg_loss:.4} lr: {scheduler.get_last_lr()[0]:.8}")
                LOGGER.info(f"[Eval] Epoch: {epoch+1} Steps: [{idx}/{len(val_loader)}] Elapse: {remain:s}")
    
    return np.concatenate(labels), np.concatenate(preds)

def train(LOGGER, fold, model, train_loader, val_loader, val_df, df_orders, epochs, checkpoint=None):
    LOGGER.info(f"========== fold: {fold} ==========")
    np.random.seed(0)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = int(CFG.EPOCHS * len(train_loader) / CFG.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizerのstateを現在のdeviceに移す。これをしないと、保存前後でdeviceの不整合が起こる可能性がある。
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        epochs =  epochs - epoch

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    # optimizer = get_optimizer(model)
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
  
    best_score = 0.
    best_epoch = 0
    start = end = time.time()
    for e in range(epochs):
        model.train()
        # tbar = tqdm(train_loader, file=sys.stdout)
        lr = adjust_lr(optimizer, e)
        
        loss_list = []
        preds = []
        labels = []

        tmp_val_df = val_df.copy()

        for idx, data in enumerate(train_loader):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            if idx % CFG.accumulation_steps == 0 or idx == len(train_loader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            # optimizer.zero_grad()
            # pred = model(inputs[0], inputs[1])

            # loss = criterion(pred, target)
            # loss.backward()
            # optimizer.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())
            
            avg_loss = np.round(np.mean(loss_list), 4)
            end = time.time()
            remain = timeSince(start, float(idx+1)/len(train_loader))
            # tbar.set_description(f"Epoch {e+1} Elapse: {remain:s} Loss: {avg_loss:.4} lr: {scheduler.get_last_lr()[0]:.8}")

            if idx % CFG.print_freq == 0 or idx == (len(train_loader)-1):
                LOGGER.info(f"[Train] Epoch {e+1} Steps: [{idx}/{len(train_loader)}] Elapse: {remain:s} Loss: {avg_loss:.4} lr: {scheduler.get_last_lr()[0]:.8}")
                if idx % 1000 == 0 or idx == (len(train_loader)-1):
                    SAVE_PATH = "/content/drive/MyDrive/ai4code_save_path/"
                    if not os.path.exists(SAVE_PATH):
                        os.makedirs(SAVE_PATH)
                    name  = CFG.BERT_PATH.replace("/", "-")
                    save_path = SAVE_PATH + f"{name}_training_state.pt"
                    torch.save({'epoch': e, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, }, save_path)
                    LOGGER.info(f"save model and optimizer: {save_path}")

        y_val, y_pred = validate(LOGGER, model, val_loader, e)
        tmp_val_df["pred"] = tmp_val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        tmp_val_df.loc[tmp_val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = tmp_val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)

        score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
        del tmp_val_df, y_dummy
        gc.collect()

        if best_score < score:
            best_score = score
            best_epoch = e
            # LOGGER.info(f'Epoch {best_epoch+1} -  Validation MSE: {mean_squared_error(y_val, y_pred):.4f}  Save best Score: {best_score:.4f} Model')
            LOGGER.info(f'Epoch {best_epoch+1} - Validation MSE: {mean_squared_error(y_val, y_pred):.4f}  L1_loss: {avg_loss:.4f}  Save best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                              'predictions': y_pred},
                              CFG.OUTPUT_DIR+f"{CFG.BERT_PATH.replace('/', '-')}_fold{fold}.pth")
        else:
            # LOGGER.info(f'Epoch {e+1} - Validation MSE: {mean_squared_error(y_val, y_pred):.4f}  Score: {score:.4f} Model')
            LOGGER.info(f'Epoch {e+1} - Validation MSE: {mean_squared_error(y_val, y_pred):.4f}  L1_loss: {avg_loss:.4f}  Score: {score:.4f} Model')

    torch.cuda.empty_cache()
    return model, y_pred