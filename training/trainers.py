import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from transformers import Trainer

from dataloaders.imbalanced import ImbalancedDatasetSampler


class BalancedWeightUpdateTrainer(Trainer):
    def __init__(self, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = torch.tensor(weights).cuda()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
#        if model.training:
#            loss_fct = nn.CrossEntropyLoss(weight=self.weights)
#        else:
#            loss_fct = nn.CrossEntropyLoss()
        loss_fct = nn.CrossEntropyLoss(weight=self.weights)
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    # def get_train_dataloader(self) -> DataLoader:
    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")
    #     train_sampler = self._get_train_sampler()
    #
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.args.train_batch_size,
    #         sampler=ImbalancedDatasetSampler(self.train_dataset),
    #         collate_fn=self.data_collator,
    #         drop_last=self.args.dataloader_drop_last,
    #         num_workers=self.args.dataloader_num_workers,
    #     )