import pandas as pd
import numpy as np
from datasets import load_metric
from scipy.special import softmax

from modeling.models import GenericModel
from utils import config
from pycaret.classification import load_model

metric = load_metric("f1")


def compute_metric(eval_pred):
	logits, labels = eval_pred
	softmaxed = softmax(logits)
	predictions = np.argmax(logits, axis=1)
	f1_score = metric.compute(predictions=predictions, references=labels, average="weighted")
	multi_class_loss = 0
	for idx, label in enumerate(labels):
		multi_class_loss += np.log(softmaxed[idx][label])
	f1_score["multi_class_loss"] = - (multi_class_loss / len(labels))
	return f1_score


def multi_class_log_loss(predictions, actual):
	overall_sum = 0
	for i in range(0, predictions.shape[0]):
		for j in range(0, predictions.shape[1]):
			multiplier = 1 if actual.iloc[i] == j else 0
			overall_sum += multiplier * np.log(max(min(predictions[i, j], 1.0 - 10**-15), 10**-15))
	return overall_sum * -(1/predictions.shape[0])


def eval_model_bert(test_model: GenericModel):
	_, val_dataset = config.load_bert_train_val()
	targets = val_dataset["label"]
	predictions = test_model.underlying_model.decision_function(val_dataset[val_dataset.columns[0:-1]].values)
	return compute_metric((predictions, targets))


if __name__ == "__main__":
	loaded_model = load_model("/home/byrdofafeather/ByrdOfAFeather/SSGOGETA/modeling/basic_models/test")
	model = GenericModel(loaded_model.steps[-1][1])
	print(eval_model_bert(model))
