# ====================================================
# CFG
# ====================================================
class CFG:
    num_workers=4
    path="../input/my-sample-trained-models/"
    emsanble1_path="../input/pppm-deberta-v3-large-baseline-w-w-b-train/"
    emsanble2_path="../input/debertav3base0527/"
    emsanble3_path="../input/trained-models-dev3l-clr/content/pppm_outputs/"
    emsanble4_path="../input/trained-model-roberta-base-f5e4/"
    emsanble5_path="../input/trained-model-roberta-large-f5e4/"
    config_path=path+'config.pth'
    config_emsanble1_path = emsanble1_path+'config.pth'
    config_emsanble2_path = emsanble2_path+'config.pth'
    config_emsanble3_path = emsanble3_path+'config.pth'
    config_emsanble4_path = emsanble4_path+'config.pth'
    config_emsanble5_path = emsanble5_path+'config.pth'
    model="microsoft/deberta-v3-large"
    emsanble2_model="microsoft/deberta-v3-base"
    emsanble4_model="roberta-base"
    emsanble5_model="roberta-large"
    batch_size=32
    fc_dropout=0.2
    target_size=1
    max_len=133
    roberta_max_len=175
    seed=42
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    emsanble1_fold=4
    emsanble1_trn_fold=[0, 1, 2, 3]
    emsanble2_fold=5
    emsanble2_trn_fold=[0, 1, 2, 3, 4]
    emsanble3_fold=5
    emsanble3_trn_fold=[0, 1, 2, 3, 4]
    emsanble4_fold=5
    emsanble4_trn_fold=[0, 1, 2, 3, 4]
    emsanble5_fold=5
    emsanble5_trn_fold=[0, 1, 2, 3, 4]
    emsanble=True
    fix_score=False