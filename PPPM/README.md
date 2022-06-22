# kaggle competition：PPPM
LB：0.8415<br>
PB：0.8558<br>
final place：619 / 1975<br>

## use models
* deberta-v3-large
* deberta-v3-base
* deberta-large
* roberta-large
* roberta-base
* final submission is emsanbled these inference results


## About train
* training scripts
    * config.py：setting parameters
    * Utils.py：Utilities functions
    * my_Dataset.py：to make train dataset functions
    * my_model.py：custom model using huggingface transformers
    * helper_functions.py：functions to help training
    * train.py：equal train main

## About inference
* inference scripts
    * config,py：setting parameters
    * Utils.py：Utilities functions
    * my_Dataset.py：to make test dataset functions
    * my_model.py：custom model using huggingface transformers
    * inference.py：qeual inference main
