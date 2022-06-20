from pycaret.classification import *
from utils import config

bert_dataset_train, bert_dataset_val = config.load_bert_train_val(filter_adequate=False)
setup = setup(bert_dataset_train, target="target", test_data=bert_dataset_val, preprocess=False, session_id=225530, html=False, profile=False)
best = compare_models()
evaluate_model(best)
