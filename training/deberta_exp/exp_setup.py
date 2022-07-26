from transformers import TrainingArguments

from modeling.models import BertClassifier, DebertClassifier
from training.train_torch import tokenize, train, load_train_val_huggingface
from utils.config import LOG_PATH
import shutil

grids = {
    'lr': [5e-6],
    'max_length': [128, 256, 512],
    "batch_size": [32, 16],
    "model_index": [-1, 50, 75],
}


def train_baseline():
    prev_name = ""
    prev_best_eval = 0

    for lr in grids["lr"]:
        for max_length in grids["max_length"]:
            for batch_size in grids["batch_size"]:
                for model_index in grids["model_index"]:
                    model = DebertClassifier(weight_grad_index=model_index,)
                    name_format = f"lr={lr}-max_length={max_length}-model_index={model_index}-batch_size={batch_size}"
                    print(f"NOW DOING : {name_format}")
                    training_args = TrainingArguments(output_dir=f"{LOG_PATH}/{name_format}/saved_weights",per_device_train_batch_size=batch_size, evaluation_strategy="steps", num_train_epochs=3, seed=225530, eval_steps=200,run_name=f"{name_format}", save_total_limit=1,metric_for_best_model="eval_loss",report_to=["mlflow", "tensorboard"],logging_dir=f"{LOG_PATH}/{name_format}", logging_steps=200,auto_find_batch_size=False,learning_rate=lr, greater_is_better=False, save_steps=200)


                    training_args.load_best_model_at_end = True
                    dataset = load_train_val_huggingface(balanced=True)
                    tokenized = dataset.map(lambda x: tokenize(x, max_length=max_length), batched=True)
                    train_set = tokenized["train"].shuffle(seed=225530)
                    val_set = tokenized["test"].shuffle(seed=225530)
                    eval_results = train(model.underlying_model, train_set, val_set, weights_list=[1.0, 1.0, 1.0], training_args=training_args, name_format=name_format)
                    if prev_name:
                        if eval_results["eval_f1"] > prev_best_eval:
                            shutil.rmtree(f"/ssd-playpen/byrdof/SSGOGETADATA/logs/{prev_name}")
                            prev_best_eval=eval_results["eval_f1"]
                            prev_name=name_format
                        else:
                            shutil.rmtree(f"/ssd-playpen/byrdof/SSGOGETADATA/logs/{name_format}")
                            prev_name=prev_name
                    else:
                        prev_name=name_format

train_baseline()


