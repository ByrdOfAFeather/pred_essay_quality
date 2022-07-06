from typing import List

import torch
from transformers import TrainingArguments

from modeling.models import BertClassifier, EnsembleBertClassifier
from training.train_torch import load_train_val_huggingface, tokenize, train
from utils.config import LOG_PATH


def train_binary_models(remove):
    dataset = load_train_val_huggingface(filter=remove, balanced=True)
    tokenized = dataset.map(tokenize, batched=True)
    train_set = tokenized["train"].shuffle(seed=225530)
    val_set = tokenized["test"].shuffle(seed=225530)
    weight_index = -1
    weights_list = [1.0, 1.0]
    num_labels = 2
    model_container = BertClassifier(dropout=0.1, weight_grad_index=weight_index, loss_weights=weights_list,
                                     num_labels=num_labels)
    name_format = f"ROBERTA_REMOVED_{remove}_{weight_index}_BALANCED"

    training_args = TrainingArguments(output_dir=f"{LOG_PATH}/{name_format}/saved_weights",
                                      per_device_train_batch_size=8,
                                      evaluation_strategy="steps", num_train_epochs=3, seed=225530, eval_steps=500,
                                      run_name=f"{name_format}", save_total_limit=5,
                                      metric_for_best_model="eval_loss", report_to=["mlflow", "tensorboard"],
                                      logging_dir=f"{LOG_PATH}/{name_format}", logging_steps=100,
                                      auto_find_batch_size=True, learning_rate=4.961e-6)

    training_args.load_best_model_at_end = True

    final_model = train(model_container, train_set, val_set, weights_list=weights_list, training_args=training_args)


def train_ensemble(binary_models: List[str]):
    binary_berts = []
    for model in binary_models:
        local_model = BertClassifier(dropout=0.1, weight_grad_index=-1, loss_weights=[1.0,1.0], num_labels=2)
        state_dict = torch.load(model)
        local_model.load_state_dict(state_dict)
        binary_berts.append(local_model.underlying_model)

    model = EnsembleBertClassifier(model_list=binary_berts)

    name_format = "ENSEMBLE_DEBUGGING"
    training_args = TrainingArguments(output_dir=f"{LOG_PATH}/{name_format}/saved_weights",
                                      per_device_train_batch_size=8,
                                      evaluation_strategy="steps", num_train_epochs=3, seed=225530, eval_steps=500,
                                      run_name=f"{name_format}", save_total_limit=5,
                                      metric_for_best_model="eval_loss", report_to=["mlflow", "tensorboard"],
                                      logging_dir=f"{LOG_PATH}/{name_format}", logging_steps=100,
                                      auto_find_batch_size=True, learning_rate=4.961e-6)

    training_args.load_best_model_at_end = True
    dataset = load_train_val_huggingface(balanced=True)
    tokenized = dataset.map(tokenize, batched=True)
    train_set = tokenized["train"].shuffle(seed=225530)
    val_set = tokenized["test"].shuffle(seed=225530)
    final_model = train(model, train_set, val_set, weights_list=[1.0, 1.0, 1.0], training_args=training_args)

if __name__ == "__main__":
    # for exp_type in ["effective", "ineffective", "adequate"]:
    #     train_binary_models(remove=exp_type)
    train_ensemble([""])