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
	def __init__(self, dropout):
		super(BertClassifier, self).__init__()
		config = AutoConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=dropout, num_labels=3)
		self.underlying_model = AutoModel.from_pretrained("bert-base-uncased", config=config)
		idx = 0
		for i in self.underlying_model.parameters():
			idx += 1
			i.requires_grad = False
			if idx == 100:
				break
		self.classifier = nn.Linear(768, 3)
		self.loss_fct = nn.CrossEntropyLoss()
		self.num_labels = 3

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
				return_dict: Optional[bool] = None, ):
		embeddings = self.underlying_model( input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,)
		logits = self.classifier(embeddings[0][:, 0, :])
		return SequenceClassifierOutput(
            loss=self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1)),
            logits=logits,
            hidden_states=embeddings.hidden_states,
            attentions=embeddings.attentions,
        )


class DebertClassifier(GenericModel):
	def __init__(self):
		underlying_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=3)
		super(DebertClassifier, self).__init__(underlying_model)

	def forward(self, **kwargs):
		self.underlying_model(**kwargs)
