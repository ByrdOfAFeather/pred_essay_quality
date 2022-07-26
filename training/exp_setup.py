from transformers import TrainingArguments

from modeling.models import BertClassifier
from training.train_torch import tokenize, train, load_train_val_huggingface
from utils.config import LOG_PATH

grids = {
    'lr': [5e-6, 5e-5, 5e-4, 5e-3],
    'max_length': [128, 256, 512],
    "batch_size": [32, 16, 8],
    "model_index": [-1, 50, 75, 100, 125, 150, 175],
}


def train_baseline():
    for lr in grids["lr"]:
        for max_length in grids["max_length"]:
            # for batch_size in grids["batch_size"]:
            for model_index in grids["model_index"]:
                dataset = load_train_val_huggingface(balanced=True)
                tokenized = dataset.map(lambda x: tokenize(x, max_length=max_length), batched=True)
                train_set = tokenized["train"].shuffle(seed=225530)
                discourse_types = set(train_set("discourse_type"))
                print(f"THIS IS LEN DISCOURSETYPES: {len(discourse_types)}")
                val_set = tokenized["test"].shuffle(seed=225530)
                model = BertClassifier(dropout=0.1, weight_grad_index=model_index, loss_weights=[1.0, 1.0, 1.0],
                                       num_labels=3, discourse_types=discourse_types)
                name_format = f"lr = {lr} max_length={max_length} model_index={model_index}"
                print(f"NOW DOING : {name_format}")
                training_args = TrainingArguments(output_dir=f"{LOG_PATH}/{name_format}/saved_weights",
                                                  per_device_train_batch_size=8,
                                                  evaluation_strategy="steps", num_train_epochs=3, seed=225530,
                                                  eval_steps=200,
                                                  run_name=f"{name_format}", save_total_limit=5,
                                                  metric_for_best_model="eval_loss",
                                                  report_to=["mlflow", "tensorboard"],
                                                  logging_dir=f"{LOG_PATH}/{name_format}", logging_steps=100,
                                                  auto_find_batch_size=True,
                                                  learning_rate=lr
                                                  )

                training_args.load_best_model_at_end = True

                final_model = train(model, train_set, val_set, weights_list=[1.0, 1.0, 1.0],
                                    training_args=training_args)


train_baseline()
