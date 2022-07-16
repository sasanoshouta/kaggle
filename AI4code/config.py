# ====================================================
# Config
# ====================================================
class CFG:
    # BERT_PATH = "distilbert-base-uncased"
    BERT_PATH = "microsoft/codebert-base"
    # BERT_PATH = "bert-large-uncased"
    DATA_PATH = "/content/ai4code"
    OUTPUT_DIR = "./ai4code_outputs/"
    MAX_LEN = 512
    NUM_TRAIN = 139256 # Max Length = 139256
    NVALID = 0.1  # size of validation set
    BS = 8
    NW = 4
    EPOCHS = 5
    N_FOLD = 10
    upload_from_colab = True
    scheduler = "cosine" # ['linear', 'cosine']
    num_warmup_steps=0
    gradient_accumulation_steps = 1
    num_cycles=0.5 # 0.5
    batch_scheduler=True
    apex=True
    accumulation_steps = 4
    total_max_len = 512
    print_freq=100
    # SAVED_MODEL_PATH = "/content/drive/MyDrive/ai4code_save_path/distilbert-base-uncased_training_state.pt"
    SAVED_MODEL_PATH = "/content/drive/MyDrive/ai4code_save_path/microsoft-codebert-base_training_state.pt"
    # SAVED_MODEL_PATH = "/content/drive/MyDrive/ai4code_save_path/bert-large-uncased_training_state.pt"
    TO_BE_CONTENUE = False