import numpy as np
import torch.nn
from scipy.special import softmax
from datasets import load_metric
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModel, AutoConfig, \
	AutoModelForSequenceClassification
from utils import config
from modeling.models import BertClassifier, DebertClassifier
from utils.eval_model import compute_metric

metric = load_metric("f1")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)


def tokenize(x):
	return tokenizer(x["discourse_text"], padding="max_length", truncation=True)


def train():
	# configer = AutoConfig.from_pretrained("/home/byrdofafeather/ByrdOfAFeather/SSGOGETA/training/test_trainer/checkpoint-29000")
	# model_container = AutoModelForSequenceClassification.from_config(configer)
	model_container = BertClassifier()
	training_args = TrainingArguments(output_dir="test_trainer", per_device_train_batch_size=5,
	                                  evaluation_strategy="steps", num_train_epochs=5, seed=225530,
	                                  run_name="bert_finetune_discourse_text_and_type",)
	dataset = config.load_train_val_huggingface()
	tokenized = dataset.map(tokenize, batched=True)
	train_set = tokenized["train"].shuffle(seed=225530)
	val_set = tokenized["test"].shuffle(seed=225530)

	trainer = Trainer(
		model=model_container.underlying_model,
		train_dataset=train_set,
		eval_dataset=val_set,
		compute_metrics=compute_metric,
		args=training_args,

	)
	trainer.train()
	print(trainer.evaluate(val_set))
	print("here")


def eval_baseline():
	train_set, val_set = config.load_bert_train_val()
	ineffective_label, effective_label, adequate_label = config.get_encoder().transform(["Ineffective", "Effective", "Adequate"])
	print(ineffective_label, effective_label, adequate_label)
	prob_adequate = len(train_set[train_set["label"] == adequate_label]) / train_set.shape[0]
	prob_effective = len(train_set[train_set["label"] == effective_label]) / train_set.shape[0]
	prob_ineffective = len(train_set[train_set["label"] == ineffective_label]) / train_set.shape[0]
	preds = [(prob_adequate, prob_effective, prob_ineffective) for _ in val_set.index]
	print(compute_metric((preds, val_set["label"])))


train()
# eval_baseline()