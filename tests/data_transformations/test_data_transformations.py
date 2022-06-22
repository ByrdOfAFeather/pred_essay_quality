import pandas as pd
from utils import config
from data_transformations.pre_processors import PreProcessor, PreProcessorMethods
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
processor = PreProcessor(tokenizer, [PreProcessorMethods.AddPosition, PreProcessorMethods.AddDiscourseType])
full_data = pd.read_csv(f"{config.DATA_PATH}/train.csv")
processor.preprocess(full_data)
print(full_data["discourse_text"])