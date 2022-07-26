import torch
from torch import nn

from data_transformations.pre_processors import PreProcessor, PreProcessorMethods
from modeling.models import BertClassifier, DebertClassifier
from training.trainers import BalancedWeightUpdateTrainer
from utils.config import get_encoder
from utils.eval_model import compute_metric
from scipy.special import softmax
from datasets import load_metric
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModel, AutoConfig, \
    AutoModelForSequenceClassification, EarlyStoppingCallback
from datasets import load_dataset
# from requirementsforcomp import BalancedWeightUpdateTrainer, PreProcessor, PreProcessorMethods, BertClassifier, \
#     compute_metric, get_encoder
import pandas as pd
import mlflow
from utils.config import DATA_PATH, LOG_PATH


def load_train_val_huggingface(filter="", balanced=False):
    if balanced:
        extra_args = "balanced"
    else:
        extra_args = ""

    print(f"Loading dataset with extra arguments {extra_args}")
    if filter == "adequate":
        files = {"train": f"train_split_no_adequate_{extra_args}.csv", "test": f"val_split_no_adequate.csv"}
    elif filter == "effective":
        files = {"train": f"train_split_no_effective_{extra_args}.csv", "test": f"val_split_no_effective.csv"}
    elif filter == "ineffective":
        files = {"train": f"train_split_no_ineffective_{extra_args}.csv", "test": f"val_split_no_ineffective.csv"}
    else:
        files = {"train": f"train_split_{extra_args}_oversample.csv", "test": "val_split.csv"}
    return load_dataset(f"{DATA_PATH}", data_files=files)


import datasets
from datasets.download.download_config import DownloadConfig

# from pycaret.classification import load_model



datasets.config.HF_MODULES_CACHE = "/ssd-playpen/byrdof/transformers"
download_config = DownloadConfig(cache_dir="/ssd-playpen/byrdof/transformers")
metric  = load_metric("f1", cache_dir="/ssd-playpen/byrdof/transformers", download_config=download_config)
tokenizer = AutoTokenizer.from_pretrained("roberta-base", model_max_length=512)


def tokenize(x, max_length):
    return tokenizer(x["discourse_text"], padding="max_length", max_length=max_length, truncation=True)


def tokenize_t(x):
    return tokenizer(x["discourse_text"], padding="max_length", max_length=400, truncation=True, return_tensors="pt")


def train(model_container, train_set, val_set, weights_list, training_args, name_format):

    trainer = BalancedWeightUpdateTrainer(
        weights=weights_list,
        model=model_container,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metric,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )
    trainer.train()
    with mlflow.start_run(run_name=f"{name_format}-EVAL"):
        results = trainer.evaluate(eval_dataset=val_set)
        mlflow.end_run()
    return results



# test_vals = pd.read_csv(f"{DATA_PATH}/test.csv")
# pre_processor = PreProcessor(tokenizer, [PreProcessorMethods.All])
# pre_processor.preprocess(test_vals)
# test_vals.to_csv("test_processed.csv", index_label="Index")
# test_vals = load_dataset(f"csv", data_files={"test": "test_processed.csv"})
# test_vals = test_vals.map(tokenize, batched=True)
# collected_results = []
# final_model.eval()
# for test_entry in test_vals["test"]:
#     test_entry = tokenize_t(test_entry)
#     model_logits = final_model(test_entry["input_ids"].cuda(), attention_mask=test_entry["attention_mask"].cuda(),
#                                token_type_ids=test_entry["token_type_ids"].cuda()).get("logits")
#     collected_results.append(softmax(model_logits.cpu().detach().numpy()[0]))
#
# out_dict = {}
# test_values = pd.read_csv(f"{DATA_PATH}/test.csv")
# for idx, res in enumerate(collected_results):
#     out_dict[test_values.iloc[idx, 0]] = res
# #
# out_frame = pd.DataFrame.from_dict(out_dict, orient="index")
# encoder = get_encoder()
# out_frame.columns = encoder.inverse_transform([0, 1, 2])
# print(out_frame)
# out_frame.to_csv("submission.csv", index=True, index_label="discourse_id")
