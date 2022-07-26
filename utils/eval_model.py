import torch
import pandas as pd
import numpy as np
from datasets import load_metric
from scipy.special import softmax
import torch.nn as nn
from modeling.models import GenericModel
from utils import config
import datasets
from datasets.download.download_config import DownloadConfig

# from pycaret.classification import load_model



datasets.config.HF_MODULES_CACHE = "/ssd-playpen/byrdof/transformers"
download_config = DownloadConfig(cache_dir="/ssd-playpen/byrdof/transformers")
metric = load_metric("f1", cache_dir="/ssd-playpen/byrdof/transformers", download_config=download_config)



def compute_metric(eval_pred):
	logits, labels = eval_pred
	softmaxed = softmax(logits)
	predictions = np.argmax(logits, axis=1)
	f1_score = metric.compute(predictions=predictions, references=labels, average="weighted")
	multi_class_loss = 0
	for idx, label in enumerate(labels):
		multi_class_loss += np.log(softmaxed[idx][label])
	f1_score["multi_class_loss"] = - (multi_class_loss / len(labels))
	loss_fct = nn.CrossEntropyLoss()
	# f1_score["non_weighted_loss"] = loss_fct(torch.tensor(logits).view(-1, 3), torch.tensor(labels).long().view(-1))
	no_acc, no_acc0, no_acc1, no_acc2 = 0, 0, 0, 0
	no_corr_acc, no_corr_acc0, no_corr_acc1, no_corr_acc2 = 0, 0, 0, 0
	for idx, label in enumerate(labels):
		if label == 0:
			if np.argmax(softmaxed[idx]) == label:
				no_corr_acc0 += 1
			no_acc0 += 1
		elif label == 1:
			if np.argmax(softmaxed[idx]) == label:
				no_corr_acc1 += 1
			no_acc1 += 1
		elif label == 2:
			if np.argmax(softmaxed[idx]) == label:
				no_corr_acc2 += 1
			no_acc2 += 1
		if np.argmax(softmaxed[idx]) == label:
			no_corr_acc += 1
		no_acc += 1

	acc = no_corr_acc / no_acc
	acc_0 = no_corr_acc0 / no_acc0
	acc_1 = no_corr_acc1 / no_acc1
	if no_acc2 == 0:
		acc_2 = 0
	else:
		acc_2 = no_corr_acc2 / no_acc2
	f1_score["accuracy"] = acc
	f1_score["accuracy_0"] = acc_0
	f1_score["accuracy_1"] = acc_1
	f1_score["accuracy_2"] = acc_2
	return f1_score


def multi_class_log_loss(predictions, actual):
	overall_sum = 0
	for i in range(0, predictions.shape[0]):
		for j in range(0, predictions.shape[1]):
			multiplier = 1 if actual.iloc[i] == j else 0
			overall_sum += multiplier * np.log(max(min(predictions[i, j], 1.0 - 10 ** -15), 10 ** -15))
	return overall_sum * -(1 / predictions.shape[0])


def eval_model_bert(test_model: GenericModel):
	_, val_dataset = config.load_bert_train_val()
	targets = val_dataset["label"]
	predictions = test_model.underlying_model.decision_function(val_dataset[val_dataset.columns[0:-1]].values)
	return compute_metric((predictions, targets))


if __name__ == "__main__":
	loaded_model = load_model("/home/byrdofafeather/ByrdOfAFeather/SSGOGETA/modeling/basic_models/test")
	model = GenericModel(loaded_model.steps[-1][1])
	print(eval_model_bert(model))
