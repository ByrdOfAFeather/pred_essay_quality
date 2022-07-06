def train_all():
    dataset = load_train_val_huggingface(filter_adequate=False)
    tokenized = dataset.map(tokenize, batched=True)
    train_set = tokenized["train"].shuffle(seed=225530)
    val_set = tokenized["test"].shuffle(seed=225530)
    weight_index = 175
    weights_list = [1.0, 1.0]
    num_labels = 3
    model_container = BertClassifier(dropout=0.1, weight_grad_index=weight_index, loss_weights=weights_list,
                                     num_labels=2)
    state_dict = torch.load("/home/byrdofafeather/ByrdOfAFeather/SSGOGETADATA/logs/THREE_CLASS_FINED_TWO_CLASS_CLASSIFICATION_BERT_150_/saved_weights/checkpoint-2000/pytorch_model.bin")
    model_container.load_state_dict(state_dict)
    model_container.loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.3,1.0,1.0]))
    model_container.classifier = nn.Linear(768, 3)
    model_container.num_labels = 3
    final_model = train(model_container, train_set, val_set, num_labels=3)

