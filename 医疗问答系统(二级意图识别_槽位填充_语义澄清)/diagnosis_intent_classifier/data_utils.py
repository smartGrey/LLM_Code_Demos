import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def load_data(data_path):
    data_df = pd.read_csv(data_path, header=0, sep=',', names=["text", "label_class", "label_id"])
    texts = list(data_df.text)
    label_ids = list(data_df.label_id.map(int))
    return texts, label_ids
# (['肾结石，输尿管结石一般用什么药呢而且效果较好？', '婴儿会有痔疮吗', ......], [5, 1, ......])


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.texts, self.labels = load_data(data_path)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]
# dataset: [(text, label), ......]


def collate_fn(batch, config):
    inputs = config.diagnosis_intent_classifier_tokenizer.batch_encode_plus(
	    [item[0] for item in batch], # texts
	    padding='max_length',
	    truncation=True,
	    max_length=config.diagnosis_intent_classifier_max_len,
	    return_tensors='pt'
    )
    return (
	    inputs["input_ids"].to(config.device), # (batch_size, max_len)
	    inputs["attention_mask"].to(config.device), # (batch_size, max_len)
	    inputs["token_type_ids"].to(config.device), # (batch_size, max_len)
    ), torch.tensor([item[1] for item in batch]).long().to(config.device) # labels - (batch_size)


def get_dataloader(config):
    train_dataset = MyDataset(config.diagnosis_intent_classifier_train_data_path)
    test_dataset = MyDataset(config.diagnosis_intent_classifier_test_data_path)

    train_iter = DataLoader(
	    train_dataset,
	    batch_size=config.diagnosis_intent_classifier_batch_size,
	    collate_fn=lambda batch: collate_fn(batch, config),
	    drop_last=True,
	    shuffle=True
    )
    test_iter = DataLoader(
	    test_dataset,
	    batch_size=config.diagnosis_intent_classifier_batch_size,
	    collate_fn=lambda batch: collate_fn(batch, config),
	    drop_last=True,
	    shuffle=False
    )

    return train_iter, test_iter
