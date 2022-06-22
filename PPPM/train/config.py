# ====================================================
# CFG
# ====================================================
class CFG:
    wandb=False
    competition='PPPM'
    debug=False
    apex=True
    print_freq=100
    num_workers=4
    model="microsoft/deberta-v3-large"
    # model="microsoft/deberta-v3-base"
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5 # 0.5
    num_warmup_steps=0
    epochs=4
    encoder_lr=3e-5 # 2e-5
    decoder_lr=3e-5 # 2e-5
    min_lr=2e-6 # 1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=16
    fc_dropout=0.2
    target_size=1
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=False
    all_train=True
    upload_from_colab=True
    adv_lr=0.0000
    adv_eps=0.001
    
if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]