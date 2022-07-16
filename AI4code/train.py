# ====================================================
# imports
# ====================================================
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
os.system("pip install transformers")
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from config import CFG
from functions import *
from myDataset import MarkdownDataset
from myModel import *

OUTPUT_DIR = './ai4code_outputs/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if  __name__ == "__main__":
  # ====================================================
  # create Train DataFame
  # ====================================================
  LOGGER = get_logger(filename=CFG.OUTPUT_DIR+'train')

  data_dir = Path(CFG.DATA_PATH)
  paths_train = list((data_dir / 'train').glob('*.json'))[:CFG.NUM_TRAIN]
  notebooks_train = [read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')]
  df = (pd.concat(notebooks_train).set_index('id', append=True).swaplevel().sort_index(level='id', sort_remaining=False))
  # display(df)

  # Split the string representation of cell_ids into a list
  df_orders = pd.read_csv(data_dir / 'train_orders.csv',index_col='id',squeeze=True,).str.split()
  # display(df_orders)

  df_orders_ = df_orders.to_frame().join(df.reset_index('cell_id').groupby('id')['cell_id'].apply(list), how='right',)
  ranks = {}
  for id_, cell_order, cell_id in df_orders_.itertuples():
      ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

  df_ranks = (pd.DataFrame.from_dict(ranks, orient='index').rename_axis('id').apply(pd.Series.explode).set_index('cell_id', append=True))
  # display(df_ranks)

  # ====================================================
  # save Tokenizer
  # ====================================================
  tokenizer = AutoTokenizer.from_pretrained(CFG.BERT_PATH)
  tokenizer.save_pretrained(CFG.OUTPUT_DIR+'tokenizer/')

  df_ancestors = pd.read_csv(data_dir / 'train_ancestors.csv', index_col='id')

  df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])
  df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

  # ====================================================
  # CV Split：GroupShuffleSplit
  # ====================================================
  splitter = GroupShuffleSplit(n_splits=CFG.N_FOLD, test_size=CFG.NVALID, random_state=0)
  train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))
  train_df = df.loc[train_ind].reset_index(drop=True)
  val_df = df.loc[val_ind].reset_index(drop=True)

  val_fts = get_features(val_df)
  json.dump(val_fts, open("./val_fts.json","wt"))
  train_fts = get_features(train_df)
  json.dump(train_fts, open("./train_fts.json","wt"))

  del df
  gc.collect()

  del df_orders_, ranks, df_ranks, notebooks_train, paths_train
  gc.collect()

  # ====================================================
  # Train
  # Model：DistilBertModel
  # 次やってみたいモデル：bert-large-uncased, microsoft/codebert-base
  # ====================================================
  LOGGER.info(f"CFG.BERT_PATH: {CFG.BERT_PATH}")
  LOGGER.info(f"CFG.MAX_LEN : {CFG.MAX_LEN}")
  LOGGER.info(f"CFG.total_max_len : {CFG.total_max_len}")
  LOGGER.info(f"CFG.NUM_TRAIN : {CFG.NUM_TRAIN}")
  LOGGER.info(f"CFG.NVALID : {CFG.NVALID}")
  LOGGER.info(f"CFG.BS : {CFG.BS}")
  LOGGER.info(f"CFG.EPOCHS : {CFG.EPOCHS}")

  # ====================================================
  # About Markdown DataLoad
  # ====================================================
  train_df_mark = train_df[train_df["cell_type"] == "markdown"].reset_index(drop=True)
  val_df_mark = val_df[val_df["cell_type"] == "markdown"].reset_index(drop=True)

  train_fts = json.load(open(data_dir / "train_fts.json"))
  val_fts = json.load(open(data_dir / "val_fts.json"))

  train_ds = MarkdownDataset(train_df_mark, model_name_or_path=CFG.BERT_PATH, md_max_len=CFG.MAX_LEN,
                            total_max_len=CFG.total_max_len, fts=train_fts)
  val_ds = MarkdownDataset(val_df_mark, model_name_or_path=CFG.BERT_PATH, md_max_len=CFG.MAX_LEN,
                              total_max_len=CFG.total_max_len, fts=val_fts)
  train_loader = DataLoader(train_ds, batch_size=CFG.BS, shuffle=True, num_workers=CFG.NW,
                                pin_memory=False, drop_last=True)
  val_loader = DataLoader(val_ds, batch_size=CFG.BS, shuffle=False, num_workers=CFG.NW,
                              pin_memory=False, drop_last=False)

  model = MarkdownModel(CFG, trained_model_path=None)
  torch.save(model.config, CFG.OUTPUT_DIR+"config.pth")
  # ====================================================
  # Saved model and optimizer load
  # ====================================================
  if CFG.TO_BE_CONTENUE:
      print(f"CFG.TO_BE_CONTENUE: {CFG.TO_BE_CONTENUE}")
      checkpoint = torch.load(CFG.SAVED_MODEL_PATH)
      model.load_state_dict(checkpoint['model_state_dict'])
      model = model.cuda()
      model, y_pred = train(LOGGER=LOGGER, fold=0, model=model, train_loader=train_loader, val_loader=val_loader, 
                                       val_df=val_df, df_orders=df_orders, epochs=CFG.EPOCHS, checkpoint=checkpoint)
  else:
      print(f"CFG.TO_BE_CONTENUE: {CFG.TO_BE_CONTENUE}")
      model = model.cuda()
      model, y_pred = train(LOGGER=LOGGER, fold=0, model=model, train_loader=train_loader, val_loader=val_loader, 
                                       val_df=val_df, df_orders=df_orders, epochs=CFG.EPOCHS, checkpoint=None)