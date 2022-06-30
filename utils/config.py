import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset

pd.set_option('mode.chained_assignment', None)

DATA_PATH = "/home/byrdofafeather/ByrdOfAFeather/SSGOGETADATA/data"
LOG_PATH = "/home/byrdofafeather/ByrdOfAFeather/SSGOGETADATA/logs"


def load_bert_train_val(filter_adequate=False):
	full_data = pd.read_csv(f"{DATA_PATH}/train.csv")
	bert_data = pd.read_csv(f"{DATA_PATH}/bert_embeddings_train.csv", index_col="Index")
	with open(DATA_PATH + "/bert_splits.json") as f:
		data = json.load(f)
	train_targets = full_data.iloc[data["train_indicies"]]["discourse_effectiveness"]
	val_targets = full_data.iloc[data["val_indicies"]]["discourse_effectiveness"]
	bert_train = bert_data.iloc[data["train_indicies"]]
	bert_val = bert_data.iloc[data["val_indicies"]]
	encoder = get_encoder()

	bert_train["label"] = encoder.transform(train_targets)
	bert_val["label"] = encoder.transform(val_targets)

	if filter_adequate:
		bert_train.drop(bert_train[bert_train["label"] == "Adequate"].index, inplace=True)
		bert_val.drop(bert_val[bert_val["label"] == "Adequate"].index, inplace=True)

	return bert_train, bert_val


def load_train_val_huggingface():
	files = {"train": "train_split.csv", "test": "val_split.csv"}
	return load_dataset(f"{DATA_PATH}", data_files=files)


def load_bert_test_val():
	bert_data = pd.read_csv(f"{DATA_PATH}/bert_embeddings_test.csv", index_col="Index")
	return bert_data


def create_bert_train_val():
	full_data = pd.read_csv(f"{DATA_PATH}/train.csv")
	essay_ids = full_data["essay_id"].unique()
	train_essays, val_and_test_essay = train_test_split(essay_ids, test_size=.2, random_state=225530)
	val_essay, test_essay = train_test_split(val_and_test_essay, test_size=.5, random_state=225530)
	train_indicies = []
	val_indicies = []
	test_indicies = []

	def get_indicies(fill, essays):
		for ids in essays:
			fill.extend(full_data[full_data["essay_id"] == ids].index)

	get_indicies(train_indicies, train_essays)
	get_indicies(val_indicies, val_essay)
	get_indicies(test_indicies, test_essay)

	print(len(train_indicies) / full_data.shape[0])
	print(len(val_indicies) / full_data.shape[0])
	print(len(test_indicies) / full_data.shape[0])

	output = {
		"train_indicies": list(train_indicies),
		"val_indicies": list(val_indicies),
		"test_indicies": list(test_indicies)
	}

	with open(DATA_PATH + "/bert_splits.json", "w") as f:
		json.dump(output, f)


def get_encoder():
	encoder = LabelEncoder()
	encoder.fit(["Ineffective", "Adequate", "Effective"])
	return encoder


if __name__ == "__main__":
	create_bert_train_val()
