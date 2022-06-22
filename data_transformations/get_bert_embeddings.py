import math
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

from utils import config

HOME_DIRECTORY = config.DATA_PATH


def get_combined_text_and_discourse_type(text: str, discourse_type: str):
    return text + "|" + discourse_type


def get_embeddings(dataset: str, output_file_name: str, tokenizer, model):
    t_data = pd.read_csv(dataset)
    no_iter = math.ceil(t_data.shape[0] / 6)
    texts = t_data.apply(lambda x: get_combined_text_and_discourse_type(x["discourse_text"], x["discourse_type"]), axis=1).values
    print(texts)
    cls_dataframe = pd.DataFrame()
    for idx in range(no_iter):
        if idx % 10 == 0:
            print(f"{idx} / {no_iter}")
        start_idx = 6 * idx
        encodings = tokenizer([i for i in list(texts[start_idx:start_idx + 6])], padding=True, truncation=True,
                              return_tensors="pt")
        for key, value in encodings.items():
            encodings[key] = encodings[key].cuda()
        embeddings = model(**encodings)
        del encodings
        torch.cuda.empty_cache()
        cls_tokens = embeddings[0][:, 0, :].detach().cpu().numpy()
        del embeddings
        torch.cuda.empty_cache()
        local_dataframe = pd.DataFrame(cls_tokens)
        # It's probably more efficient to concat all at the end
        cls_dataframe = pd.concat((cls_dataframe, local_dataframe), ignore_index=True)

    cls_dataframe.to_csv(output_file_name, index_label="Index")


def get_bert_embeddings(dataset: str, output_file_name: str):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.cuda()
    get_embeddings(dataset, output_file_name, tokenizer, model)


def get_t5_embeddings(dataset: str, output_file_name: str):
    tokenizer = BertTokenizer.from_pretrained("t5-base-uncased")
    model = BertModel.from_pretrained("t5-base-uncased")
    model.cuda()
    get_embeddings(dataset, output_file_name, tokenizer, model)


if __name__ == "__main__":
    get_embeddings(f"{config.DATA_PATH}/train.csv", "bert_embeddings_train_included_discourse_type.csv")
