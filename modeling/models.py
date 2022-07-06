import pandas as pd
import scipy
from sklearn.linear_model import RidgeClassifier
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoModel
import torch.nn as nn
import torch
from typing import Optional

from transformers.modeling_outputs import SequenceClassifierOutput


class GenericModel:
	def __init__(self, underlying_model):
		self.underlying_model = underlying_model

	def get_probabilities(self, dataset: pd.DataFrame):
		if type(self.underlying_model) == RidgeClassifier:
			return scipy.special.softmax(self.underlying_model.decision_function(dataset))
		if type(self.underlying_model) == BertClassifier:
			pass


class BertClassifier(nn.Module):
	def __init__(self, dropout, weight_grad_index, loss_weights, num_labels):
		super(BertClassifier, self).__init__()
		config = AutoConfig.from_pretrained("roberta-base", hidden_dropout_prob=dropout, num_labels=num_labels,)
		self.underlying_model = AutoModel.from_pretrained("roberta-base", config=config)
		print(f"Using {weight_grad_index} / {len(list(self.underlying_model.parameters()))}")
		if weight_grad_index != -1:
			idx = 0
			for i in self.underlying_model.parameters():
				if idx >= weight_grad_index:
					if idx < 100:
						# Don't initilize too early of layers
						idx += 1
						continue
					try:
						torch.nn.init.xavier_uniform_(i)
					except ValueError:
						pass
				else:
					idx += 1
					i.requires_grad = False

		sumer = 0
		for param in self.underlying_model.parameters():
			if not param.requires_grad: continue
			try:
				sumer += param.shape[0] * param.shape[1]
			except Exception:
				sumer += param.shape[0]

		sumer += 768 * num_labels
		print(f"No trainable params: {sumer}")
		self.classifier = nn.Linear(768, num_labels)
		self.loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(loss_weights))
		self.num_labels = num_labels

	def forward(self,
	            input_ids: Optional[torch.Tensor] = None,
	            attention_mask: Optional[torch.Tensor] = None,
	            token_type_ids: Optional[torch.Tensor] = None,
	            position_ids: Optional[torch.Tensor] = None,
	            head_mask: Optional[torch.Tensor] = None,
	            inputs_embeds: Optional[torch.Tensor] = None,
	            labels: Optional[torch.Tensor] = None,
	            output_attentions: Optional[bool] = None,
	            output_hidden_states: Optional[bool] = None,
	            return_dict: Optional[bool] = None, **kwargs):
		embeddings = self.underlying_model(input_ids, attention_mask=attention_mask,
		                                   token_type_ids=token_type_ids,
		                                   position_ids=position_ids,
		                                   head_mask=head_mask,
		                                   inputs_embeds=inputs_embeds,
		                                   output_attentions=output_attentions,
		                                   output_hidden_states=output_hidden_states,
		                                   return_dict=return_dict, )
		logits = self.classifier(embeddings[0][:, 0, :])
		return SequenceClassifierOutput(
			loss=self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1)),
			logits=logits,
			hidden_states=embeddings.hidden_states,
			attentions=embeddings.attentions,
		)


class DebertClassifier(GenericModel):
	def __init__(self, weight_grad_index):
		self.underlying_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=3)
		idx = 0
		for i in self.underlying_model.parameters():
			if idx == weight_grad_index:
				break
			idx += 1
			i.requires_grad = False

		sumer = 0
		for param in self.underlying_model.parameters():
			if not param.requires_grad: continue
			try:
				sumer += param.shape[0] * param.shape[1]
			except Exception:
				sumer += param.shape[0]

		print(f"No trainable params: {sumer}")
		super(DebertClassifier, self).__init__(self.underlying_model)

	def forward(self, **kwargs):
		self.underlying_model(**kwargs)


class XLNetClassifier(GenericModel):
	def __init__(self, weight_grad_index):
		self.underlying_model = AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=3)
		print(f"Using {weight_grad_index} / {len(list(self.underlying_model.parameters()))}")
		idx = 0
		for i in self.underlying_model.parameters():
			if idx == weight_grad_index:
				break
			idx += 1
			i.requires_grad = False

		sumer = 0
		for param in self.underlying_model.parameters():
			if not param.requires_grad: continue
			try:
				sumer += param.shape[0] * param.shape[1]
			except Exception:
				sumer += param.shape[0]

		print(f"No trainable params: {sumer}")
		super(XLNetClassifier, self).__init__(self.underlying_model)

	def forward(self, **kwargs):
		self.underlying_model(**kwargs)