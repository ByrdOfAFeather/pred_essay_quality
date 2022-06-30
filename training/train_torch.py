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


def load_train_val_huggingface():
    files = {"train": "train_split.csv", "test": "val_split.csv"}
    return load_dataset(f"{DATA_PATH}", data_files=files)


metric = load_metric("f1")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)


def tokenize(x):
    return tokenizer(x["discourse_text"], padding="max_length", truncation=True)


def tokenize_t(x):
    return tokenizer(x["discourse_text"], padding="max_length", truncation=True, return_tensors="pt")


def train():
    # configer = AutoConfig.from_pretrained("/home/byrdofafeather/ByrdOfAFeather/SSGOGETA/training/test_trainer/checkpoint-29000")
    # model_container = AutoModelForSequenceClassification.from_config(configer)
    weight_index = 160
    # weights_list = [1.754892601431981, 3.941042476215999, 5.668144151088842]
    weights_list = [1.0, 1.0, 1.0]
    name_format = f"BERT_DATAV3_WEIGHT_INDEX_{weight_index}_REINIT_IMBALACNEDSAMPLER_INCEPOCH"
    model_container = BertClassifier(dropout=0.1, weight_grad_index=weight_index, loss_weights=weights_list)
    training_args = TrainingArguments(output_dir=f"{LOG_PATH}/{name_format}/saved_weights", per_device_train_batch_size=8,
                                      evaluation_strategy="steps", num_train_epochs=5, seed=225530, eval_steps=500,
                                      run_name=f"{name_format}", save_total_limit=5,
                                      metric_for_best_model="eval_loss", report_to=["mlflow", "tensorboard"],
                                      logging_dir=f"{LOG_PATH}/{name_format}", logging_steps=100,
                                      auto_find_batch_size=True)
    training_args.load_best_model_at_end = True

    dataset = load_train_val_huggingface()
    tokenized = dataset.map(tokenize, batched=True)
    train_set = tokenized["train"].shuffle(seed=225530)
    val_set = tokenized["test"].shuffle(seed=225530)
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


final_model = train()

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
