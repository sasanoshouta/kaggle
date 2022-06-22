# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
print(f"torch.__version__: {torch.__version__}")
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

os.system('pip uninstall -y transformers')
os.system('pip uninstall -y tokenizers')
os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels-dataset transformers')
os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels-dataset tokenizers')
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
%env TOKENIZERS_PARALLELISM=true

# ====================================================
# myself code import
# ====================================================
from config import CFG
from my_model import CustomModel
from my_Dataset import prepare_input, prepare_roberta_input, TestDataset, roberta_TestDataset
from Utils import get_score, get_logger, seed_everything

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================================
# Directory settings
# ====================================================
import os

INPUT_DIR = '../input/us-patent-phrase-to-phrase-matching/'
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOGGER = get_logger(filename=OUTPUT_DIR+'train')
seed_everything(seed=42)

# ====================================================
# Read oof file
# ====================================================
oof_df = pd.read_pickle(CFG.path+'oof_df.pkl')
labels = oof_df['score'].values
preds = oof_df['pred'].values
score = get_score(labels, preds)
LOGGER.info(f'CV Score: {score:<.4f}')

# ====================================================
# Data Loading
# ====================================================
test = pd.read_csv(INPUT_DIR+'test.csv')
submission = pd.read_csv(INPUT_DIR+'sample_submission.csv')

# ====================================================
# CPC Data
# ====================================================
cpc_texts = torch.load(CFG.path+"cpc_texts.pth")
test['context_text'] = test['context'].map(cpc_texts)

test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']

# ====================================================
# tokenizer
# ====================================================
CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path+'tokenizer/')

# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

if __name__ == '__main__':
    test_dataset = TestDataset(CFG, test)
    test_loader = DataLoader(test_dataset,
                            batch_size=CFG.batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    predictions = []
    for fold in CFG.trn_fold:
        model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
        state = torch.load(CFG.path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                          map_location=torch.device('cpu'))
        model.load_state_dict(state['model'])
        prediction = inference_fn(test_loader, model, device)
        predictions.append(prediction)
        del model, state, prediction; gc.collect()
        torch.cuda.empty_cache()
    predictions = np.mean(predictions, axis=0)

    submission['score'] = predictions
    submission[['id', 'score']].to_csv('submission.csv', index=False)

    if CFG.emsanble:
        test_dataset = TestDataset(CFG, test)
        test_loader = DataLoader(test_dataset,
                            batch_size=CFG.batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
        # ====================================================
        # emsanble1: oof
        # ====================================================
        oof_df = pd.read_pickle(CFG.emsanble1_path+'oof_df.pkl')
        labels = oof_df['score'].values
        preds = oof_df['pred'].values
        score = get_score(labels, preds)
        LOGGER.info(f'CV Score: {score:<.4f}')
        
        # ====================================================
        # emsanble1: tokenizer
        # ====================================================
        CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.emsanble1_path+'tokenizer/')
        
        predictions = []
        for fold in CFG.emsanble1_trn_fold:
            model = CustomModel(CFG, config_path=CFG.config_emsanble1_path, pretrained=False)
            state = torch.load(CFG.emsanble1_path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                              map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            prediction = inference_fn(test_loader, model, device)
            predictions.append(prediction)
            del model, state, prediction; gc.collect()
            torch.cuda.empty_cache()
        predictions = np.mean(predictions, axis=0)
        emsanble1_submission = pd.read_csv(INPUT_DIR+'sample_submission.csv')
        emsanble1_submission['score'] = predictions
        
        """
        # ====================================================
        # emsanble2: oof
        # ====================================================
        oof_df = pd.read_pickle(CFG.emsanble2_path+'oof_df.pkl')
        labels = oof_df['score'].values
        preds = oof_df['pred'].values
        score = get_score(labels, preds)
        LOGGER.info(f'CV Score: {score:<.4f}')
        
        # ====================================================
        # emsanble2: tokenizer
        # ====================================================
        CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.emsanble2_path+'tokenizer/')
        
        predictions = []
        for fold in CFG.emsanble2_trn_fold:
            model = CustomModel(CFG, config_path=CFG.config_emsanble2_path, pretrained=False)
            state = torch.load(CFG.emsanble2_path+f"{CFG.emsanble2_model.replace('/', '-')}_fold{fold}_best.pth",
                              map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            prediction = inference_fn(test_loader, model, device)
            predictions.append(prediction)
            del model, state, prediction; gc.collect()
            torch.cuda.empty_cache()
        predictions = np.mean(predictions, axis=0)
        emsanble2_submission = pd.read_csv(INPUT_DIR+'sample_submission.csv')
        emsanble2_submission['score'] = predictions
        """
        # ====================================================
        # emsanble3: oof
        # ====================================================
        oof_df = pd.read_pickle(CFG.emsanble3_path+'oof_df.pkl')
        labels = oof_df['score'].values
        preds = oof_df['pred'].values
        score = get_score(labels, preds)
        LOGGER.info(f'CV Score: {score:<.4f}')
        
        # ====================================================
        # emsanble3: tokenizer
        # ====================================================
        CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.emsanble3_path+'tokenizer/')
        
        predictions = []
        for fold in CFG.emsanble3_trn_fold:
            model = CustomModel(CFG, config_path=CFG.config_emsanble3_path, pretrained=False)
            state = torch.load(CFG.emsanble3_path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                              map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            prediction = inference_fn(test_loader, model, device)
            predictions.append(prediction)
            del model, state, prediction; gc.collect()
            torch.cuda.empty_cache()
        predictions = np.mean(predictions, axis=0)
        emsanble3_submission = pd.read_csv(INPUT_DIR+'sample_submission.csv')
        emsanble3_submission['score'] = predictions
        """
        # ====================================================
        # emsanble4: oof
        # ====================================================
        oof_df = pd.read_pickle(CFG.emsanble4_path+'oof_df.pkl')
        labels = oof_df['score'].values
        preds = oof_df['pred'].values
        score = get_score(labels, preds)
        LOGGER.info(f'CV Score: {score:<.4f}')
        
        # ====================================================
        # emsanble4: tokenizer
        # ====================================================
        CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.emsanble4_path+'tokenizer/')
        
        test_dataset = roberta_TestDataset(CFG, test)
        test_loader = DataLoader(test_dataset,
                            batch_size=CFG.batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
        
        predictions = []
        for fold in CFG.emsanble4_trn_fold:
            model = CustomModel(CFG, config_path=CFG.config_emsanble4_path, pretrained=False)
            state = torch.load(CFG.emsanble4_path+f"{CFG.emsanble4_model.replace('/', '-')}_fold{fold}_best.pth",
                              map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            prediction = inference_fn(test_loader, model, device)
            predictions.append(prediction)
            del model, state, prediction; gc.collect()
            torch.cuda.empty_cache()
        predictions = np.mean(predictions, axis=0)
        emsanble4_submission = pd.read_csv(INPUT_DIR+'sample_submission.csv')
        emsanble4_submission['score'] = predictions
        """
        # ====================================================
        # emsanble5: oof
        # ====================================================
        oof_df = pd.read_pickle(CFG.emsanble5_path+'oof_df.pkl')
        labels = oof_df['score'].values
        preds = oof_df['pred'].values
        score = get_score(labels, preds)
        LOGGER.info(f'CV Score: {score:<.4f}')
        
        # ====================================================
        # emsanble5: tokenizer
        # ====================================================
        CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.emsanble5_path+'tokenizer/')
        
        test_dataset = roberta_TestDataset(CFG, test)
        test_loader = DataLoader(test_dataset,
                            batch_size=CFG.batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
        
        predictions = []
        for fold in CFG.emsanble5_trn_fold:
            model = CustomModel(CFG, config_path=CFG.config_emsanble5_path, pretrained=False)
            state = torch.load(CFG.emsanble5_path+f"{CFG.emsanble5_model.replace('/', '-')}_fold{fold}_best.pth",
                              map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            prediction = inference_fn(test_loader, model, device)
            predictions.append(prediction)
            del model, state, prediction; gc.collect()
            torch.cuda.empty_cache()
        predictions = np.mean(predictions, axis=0)
        emsanble5_submission = pd.read_csv(INPUT_DIR+'sample_submission.csv')
        emsanble5_submission['score'] = predictions
        
        
        # ====================================================
        # create final submission
        # ====================================================
        emsanble1_submission['emsanbled_score'] = (emsanble5_submission['score'] + 
    #                                                emsanble4_submission['score'] +  
                                                  emsanble3_submission["score"] +
    #                                                emsanble2_submission["score"] + 
                                                  emsanble1_submission['score'] + 
                                                  submission["score"]) / 4
        emsanbled_submission = emsanble1_submission[["id", 'emsanbled_score']].rename(columns={"emsanbled_score": "score"})
    #     del submission,emsanble1_submission,emsanble2_submission,emsanble3_submission,emsanble4_submission,emsanble5_submission
        del submission,emsanble1_submission,emsanble3_submission,emsanble5_submission
        emsanbled_submission[['id', 'score']].to_csv('submission.csv', index=False)
