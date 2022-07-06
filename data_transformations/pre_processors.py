import enum
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import List, Any
from utils import config

VALID_DISCOURSE_TYPES = {
    'Lead', 'Position', 'Claim', 'Evidence', 'Counterclaim',
    'Rebuttal', 'Concluding Statement'
}


class PreProcessorMethods(enum.Enum):
    AddDiscourseType = "AddDiscourseType"
    AddPosition = "AddPosition"
    AddCounterClaim = "AddCounterClaim"
    AddClaim = "AddClaim"
    All = "all"


class PreProcessor:
    @staticmethod
    def add_columns(x: pd.DataFrame, tokenizer: PreTrainedTokenizer, valid_to_add_to: List[str],
                    what_to_add: str) -> Any:
        """Adds position in-place."""
        for i in valid_to_add_to:
            assert i in VALID_DISCOURSE_TYPES, "Invalid discourse to add to detected"
        assert what_to_add in VALID_DISCOURSE_TYPES, "Invalid discourse text to add"

        indexer = x["discourse_type"] == valid_to_add_to[0]
        for other_indicies in valid_to_add_to[1:]:
            indexer = (indexer) | (x["discourse_type"] == other_indicies)

        pre_add_col = x[indexer]
        for idx, pre_add in pre_add_col.iterrows():
            add_text = [i.strip() for i in
                        x[(x["discourse_type"] == what_to_add) & (x["essay_id"] == pre_add["essay_id"])][
                            "discourse_text"]]
            pre_add["discourse_text"] = pre_add[
                                            "discourse_text"].strip() + f" {tokenizer.sep_token} " + f" {tokenizer.sep_token} ".join(
                add_text)
        x[indexer] = pre_add_col

    @staticmethod
    def add_position(x: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> Any:
        return PreProcessor.add_columns(x, tokenizer, ["Lead", "Claim", "Counterclaim"], "Position")

    @staticmethod
    def add_counterclaim(x: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> Any:
        return PreProcessor.add_columns(x, tokenizer, ["Rebuttal"], "Counterclaim")

    @staticmethod
    def add_claim(x: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> Any:
        return PreProcessor.add_columns(x, tokenizer, ["Evidence", "Concluding Statement"], "Claim")

    @staticmethod
    def add_discourse_type(x: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> Any:
        """Adds discourse type in-place"""
        x["discourse_text"] = x["discourse_type"].apply(lambda x: x.strip()) + f" {tokenizer.sep_token} " + x[
            "discourse_text"].apply(lambda x: x.strip())

    def __init__(self, tokenizer, methods: List[PreProcessorMethods], encoder):
        self.tokenizer = tokenizer
        self.methods = methods
        self.encoder = encoder

        self.method_to_method = {
            PreProcessorMethods.AddPosition: self.add_position,
            PreProcessorMethods.AddDiscourseType: self.add_discourse_type,
            PreProcessorMethods.AddCounterClaim: self.add_counterclaim,
            PreProcessorMethods.AddClaim: self.add_claim
        }

    def preprocess(self, x):
        if self.methods[0] == PreProcessorMethods.All and len(self.methods) == 1:
            for method in PreProcessorMethods:
                if method == PreProcessorMethods.All: continue
                self.method_to_method[method](x, self.tokenizer)
        else:
            for method in self.methods:
                self.method_to_method[method](x, self.tokenizer)

        x["label"] = self.encoder.transform(x["discourse_effectiveness"])

        # new_discourse_text = {}
        # for idx, item in x.iterrows():
        #     with open(f"{config.DATA_PATH}/train/{item['essay_id']}.txt") as f:
        #         essay_text = f.read()
        #     new_discourse_text[idx] = item['discourse_text'] + f" {self.tokenizer.sep_token} " + essay_text
        # new_discourse_text = pd.DataFrame.from_dict(new_discourse_text, orient="index", columns=["discourse_text"])

        # x["discourse_text"] = new_discourse_text
        x["discourse_text"] = x["discourse_text"].apply(
            lambda x: x.strip().lower().replace(self.tokenizer.sep_token.lower(), self.tokenizer.sep_token).replace(
                "\n", "").replace("\r", "").replace(" ", ""))
        return x


def create_full_train_val_sets():
    full_data = pd.read_csv(f"{config.DATA_PATH}/train.csv")
    with open(config.DATA_PATH + "/bert_splits.json") as f:
        data = json.load(f)
    train = full_data.iloc[data["train_indicies"]]
    val = full_data.iloc[data["val_indicies"]]

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    pre_processor = PreProcessor(tokenizer, [PreProcessorMethods.All])
    pre_processor.preprocess(train)
    pre_processor.preprocess(val)

    train.drop(["discourse_id", "essay_id", "discourse_effectiveness"], axis=1, inplace=True)
    val.drop(["discourse_id", "essay_id", "discourse_effectiveness"], axis=1, inplace=True)
    train.to_csv(f"{config.DATA_PATH}/train_split.csv", index_label="Index")
    val.to_csv(f"{config.DATA_PATH}/val_split.csv", index_label="Index")
    print("Finished")


encoder_dict = {
    "Effective": config.get_adequate_ineffective_encoder(),
    "Ineffective": config.get_adequate_effective_encoder(),
    "Adequate": config.get_ineffective_effective_encoder(),
}

def downsample(df:pd.DataFrame, label_col_name:str) -> pd.DataFrame:
    """https://rensdimmendaal.com/data-science/undersampling-with-pandas/"""
    # find the number of observations in the smallest group
    nmin = df[label_col_name].value_counts().min()
    return (df
            # split the dataframe per group
            .groupby(label_col_name)
            # sample nmin observations from each group
            .apply(lambda x: x.sample(nmin))
            # recombine the dataframes
            .reset_index(drop=True)
            )


def create_binary_sets(remove):
    full_data = pd.read_csv(f"{config.DATA_PATH}/train.csv")
    with open(config.DATA_PATH + "/bert_splits.json") as f:
        data = json.load(f)

    train = full_data.iloc[data["train_indicies"]]
    print(f"before: {train.shape}")
    train.drop(train[train["discourse_effectiveness"] == remove].index, inplace=True)
    print(f"after: {train.shape}")
    val = full_data.iloc[data["val_indicies"]]
    print(f"before: {val.shape}")
    val.drop(val[val["discourse_effectiveness"] == remove].index, inplace=True)
    print(f"after: {val.shape}")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    pre_processor = PreProcessor(tokenizer, [PreProcessorMethods.All], encoder=encoder_dict[remove])
    pre_processor.preprocess(train)
    pre_processor.preprocess(val)

    train.drop(["discourse_id", "essay_id", "discourse_effectiveness"], axis=1, inplace=True)
    val.drop(["discourse_id", "essay_id", "discourse_effectiveness"], axis=1, inplace=True)

    train_balanced = downsample(train, "label")
    val_balanced = downsample(val, "label")
    print(f"Train Balanced Shape: {train_balanced.shape}")
    print(f"Val Balanced Shape: {val_balanced.shape}")

    train.to_csv(f"{config.DATA_PATH}/train_split_no_{remove.lower()}.csv", index_label="Index")
    val.to_csv(f"{config.DATA_PATH}/val_split_no_{remove.lower()}.csv", index_label="Index")
    train_balanced.to_csv(f"{config.DATA_PATH}/train_split_no_{remove.lower()}_balanced.csv", index_label="Index")
    val_balanced.to_csv(f"{config.DATA_PATH}/val_split_no_{remove.lower()}_balanced.csv", index_label="Index")

    train_balanced["label"] = np.random.permutation(train_balanced["label"].values)
    train_balanced.to_csv(f"{config.DATA_PATH}/train_split_no_{remove.lower()}_shuffled_labels.csv", index_label="Index")
    print("Finished")


if __name__ == "__main__":
    print("Creating datasets")
    create_binary_sets("Effective")
    # create_binary_sets("Ineffective")
    # create_binary_sets("Adequate")


