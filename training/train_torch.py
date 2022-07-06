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
    AutoModelForSequenceClassification
from datasets import load_dataset
# from requirementsforcomp import BalancedWeightUpdateTrainer, PreProcessor, PreProcessorMethods, BertClassifier, \
#     compute_metric, get_encoder
import pandas as pd
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
        files = {"train": "train_split.csv", "test": "val_split.csv"}
    return load_dataset(f"{DATA_PATH}", data_files=files)


metric = load_metric("f1")
tokenizer = AutoTokenizer.from_pretrained("roberta-base", model_max_length=512)


def tokenize(x):
    return tokenizer(x["discourse_text"], padding="max_length", truncation=True)


def tokenize_t(x):
    return tokenizer(x["discourse_text"], padding="max_length", truncation=True, return_tensors="pt")


def train(model_container, train_set, val_set, weights_list, name_format):
    # weights_list = [1.754892601431981, 3.941042476215999, 5.668144151088842]
    # weights_list = [0.3, 1.0, 1.0]
    # weights_list = [1.0, 2.0]
    training_args = TrainingArguments(output_dir=f"{LOG_PATH}/{name_format}/saved_weights", per_device_train_batch_size=8,
                                      evaluation_strategy="steps", num_train_epochs=10, seed=225530, eval_steps=500,
                                      run_name=f"{name_format}", save_total_limit=5,
                                      metric_for_best_model="eval_loss", report_to=["mlflow", "tensorboard"],
                                      logging_dir=f"{LOG_PATH}/{name_format}", logging_steps=100,
                                      auto_find_batch_size=True, learning_rate=4.961e-6)
    training_args.load_best_model_at_end = True


    trainer = BalancedWeightUpdateTrainer(
        weights=weights_list,
        model=model_container,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metric,
        args=training_args,

    )
    trainer.train()
    trainer.evaluate(eval_dataset=val_set)
    return model_container


def train_binary(remove):
    dataset = load_train_val_huggingface(filter=remove, balanced=True)
    tokenized = dataset.map(tokenize, batched=True)
    train_set = tokenized["train"].shuffle(seed=225530)
    val_set = tokenized["test"].shuffle(seed=225530)
    weight_index = -1
    weights_list = [1.0, 1.0]
    num_labels=2
    model_container = BertClassifier(dropout=0.1, weight_grad_index=weight_index, loss_weights=weights_list, num_labels=num_labels)
    name_format = f"ROBERTA_REMOVED_{remove}_{weight_index}_BALANCED"
    final_model = train(model_container, train_set, val_set, weights_list=weights_list, name_format=name_format)


def train_all():
    dataset = load_train_val_huggingface(filter_adequate=False)
    tokenized = dataset.map(tokenize, batched=True)
    train_set = tokenized["train"].shuffle(seed=225530)
    val_set = tokenized["test"].shuffle(seed=225530)
    weight_index = 175
    weights_list = [1.0, 1.0]
    num_labels = 3
    model_container = BertClassifier(dropout=0.1, weight_grad_index=weight_index, loss_weights=weights_list,
                                     num_labels=2)
    state_dict = torch.load("/home/byrdofafeather/ByrdOfAFeather/SSGOGETADATA/logs/THREE_CLASS_FINED_TWO_CLASS_CLASSIFICATION_BERT_150_/saved_weights/checkpoint-2000/pytorch_model.bin")
    model_container.load_state_dict(state_dict)
    model_container.loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.3,1.0,1.0]))
    model_container.classifier = nn.Linear(768, 3)
    model_container.num_labels = 3
    final_model = train(model_container, train_set, val_set, num_labels=3)


if __name__ == "__main__":
    for exp_type in ["effective", "ineffective", "adequate"]:
        train_binary(remove=exp_type)
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
