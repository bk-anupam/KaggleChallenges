import math
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup

# Optimizer
# Split weights in two groups, one with weight decay and the other not.
def get_optimizer(model, model_hparams):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": model_hparams["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=model_hparams["learning_rate"])

def get_lr_scheduler(optimizer, dl_train, Config):
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(dl_train) / Config.GRADIENT_ACCUMULATION_STEPS)
    num_train_steps = Config.NUM_EPOCHS * num_update_steps_per_epoch
    print(f"num_update_steps_per_epoch = {num_update_steps_per_epoch}")
    print(f"num_train_steps = {num_train_steps}")
    lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=Config.MODEL_HPARAMS["warmup_steps"],
            num_training_steps=num_train_steps,
        )
    return lr_scheduler        