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
	def __init__(self, dropout, weight_grad_index, loss_weights, num_labels, discourse_types):
		super(BertClassifier, self).__init__()
		config = AutoConfig.from_pretrained("roberta-base", hidden_dropout_prob=dropout, num_labels=num_labels,)
		self.underlying_models = {discourse_type: AutoModel.from_pretrained("roberta-base", config=config) for discourse_type in discourse_types}
		self.underlying_classifier = {discourse_type:  nn.Linear(768, num_labels) for discourse_type in discourse_types}
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
	            return_dict: Optional[bool] = None, discourse_type=None, **kwargs):
		model = self.underlying_models[discourse_type]
		embeddings = model(input_ids, attention_mask=attention_mask,
		                                   token_type_ids=token_type_ids,
		                                   position_ids=position_ids,
		                                   head_mask=head_mask,
		                                   inputs_embeds=inputs_embeds,
		                                   output_attentions=output_attentions,
		                                   output_hidden_states=output_hidden_states,
		                                   return_dict=return_dict,)
		logits = self.underlying_classifier[discourse_type](embeddings[0][:, 0, :])
		return SequenceClassifierOutput(
			loss=self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1)),
			logits=logits,
			hidden_states=embeddings.hidden_states,
			attentions=embeddings.attentions,
		)


class EnsembleBertClassifier(nn.Module):
	def __init__(self, model_list, **kwargs):
		super().__init__()
		self.model_list = model_list
		for model in self.model_list:
			# model.eval()
			model.cuda()
			idx = 0
			for parameter in model.parameters():
				if idx >= 150:
					try:
						torch.nn.init.xavier_uniform_(parameter)
					except ValueError:
						pass
				else:
					parameter.requires_grad = False
				idx += 1

		sumer = 0
		for model in self.model_list:
			for param in model.parameters():
				if not param.requires_grad: continue
				try:
					sumer += param.shape[0] * param.shape[1]
				except Exception:
					sumer += param.shape[0]

		sumer += 768 * 3*3
		print(f"No trainable params: {sumer}")

		self.classification_method = kwargs.get("classificaiton_method", "mean_pool")
		if self.classification_method == "mean_pool":
			self.classifier = nn.Linear(768, 3)
		elif self.classification_method == "all":
			self.classifier = nn.Linear(768*3, 3)

		if loss_weights := kwargs.get("loss_weights"):
			self.loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(loss_weights))
		else:
			self.loss_fct = nn.CrossEntropyLoss()

		self.num_labels = 3

	def forward(self, input_ids: Optional[torch.Tensor] = None,
	            attention_mask: Optional[torch.Tensor] = None,
	            token_type_ids: Optional[torch.Tensor] = None,
	            position_ids: Optional[torch.Tensor] = None,
	            head_mask: Optional[torch.Tensor] = None,
	            inputs_embeds: Optional[torch.Tensor] = None,
	            labels: Optional[torch.Tensor] = None,
	            output_attentions: Optional[bool] = None,
	            output_hidden_states: Optional[bool] = None,
	            return_dict: Optional[bool] = None, **kwargs):

		embeddings = torch.zeros([3, input_ids.shape[0], 768]).cuda()
		for idx, model in enumerate(self.model_list):
			embedding = model(input_ids, attention_mask=attention_mask,
											   token_type_ids=token_type_ids,
											   position_ids=position_ids,
											   head_mask=head_mask,
											   inputs_embeds=inputs_embeds,
											   output_attentions=output_attentions,
											   output_hidden_states=output_hidden_states,
											   return_dict=return_dict, )
			embeddings[idx] = embedding[0][:, 0, :]

		if self.classification_method == "all":
			logits = self.classifier(embeddings.reshape((input_ids.shape[0], 768*3)))
		else:
			logits = self.classifier(embeddings.mean(dim=0))

		return SequenceClassifierOutput(
				loss=self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1)),
				logits=logits,
				# hidden_states=embeddings.hidden_states,
				# attentions=embeddings.attentions,
			)


class DebertClassifier(GenericModel):
	def __init__(self, weight_grad_index):
		self.underlying_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=3)
		if weight_grad_index != -1:
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