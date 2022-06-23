import torch.nn as nn
import torch
from transformers import Trainer


class BalancedWeightUpdateTrainer(Trainer):
    def __init__(self, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = torch.tensor(weights).cuda()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.CrossEntropyLoss(weight=self.weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
