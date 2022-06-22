import pandas as pd
import scipy
from sklearn.linear_model import RidgeClassifier
from transformers import AutoModelForSequenceClassification, AutoConfig


class GenericModel:
	def __init__(self, underlying_model):
		self.underlying_model = underlying_model

	def get_probabilities(self, dataset: pd.DataFrame):
		if type(self.underlying_model) == RidgeClassifier:
			return scipy.special.softmax(self.underlying_model.decision_function(dataset))
		if type(self.underlying_model) == BertClassifier:
			pass


class BertClassifier(GenericModel):
	def __init__(self):
		config = AutoConfig.from_pretrained("bert-base-uncased")
		config["dropout"] = 0.4
		underlying_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3, model_args={"config":config})
		super(BertClassifier, self).__init__(underlying_model)

	def forward(self, **kwargs):
		self.underlying_model(**kwargs)

class DebertClassifier(GenericModel):
	def __init__(self):
		underlying_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=3)
		super(DebertClassifier, self).__init__(underlying_model)

	def forward(self, **kwargs):
		self.underlying_model(**kwargs)