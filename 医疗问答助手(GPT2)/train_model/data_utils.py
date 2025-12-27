import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils


# 处理原始语料
# 先拼接："[CLS]sentence1[SEP]sentence2[SEP]sentence3[SEP]"
# 再转为token id：[1,345,35,66.......]
def data_preprocess(config):
    def process(source_path, target_path):
        with open(source_path, 'rb') as f:
            data = f.read().decode("utf-8")
        qa_text_list = data.split("\n\n")

        qa_tokens_list = []
        for qa_text in qa_text_list:
            sentences = qa_text.split("\n") # 按行分割
            gpt_input_token_ids = [config.bert_tokenizer.cls_token_id] # 添加开头的[CLS]
            for sentence in sentences:
                gpt_input_token_ids += config.bert_tokenizer.encode(sentence, add_special_tokens=False)
                gpt_input_token_ids.append(config.bert_tokenizer.sep_token_id) # 每句话后面添加[SEP]
            qa_tokens_list.append(gpt_input_token_ids)

        with open(target_path, "wb") as f:
            pickle.dump(qa_tokens_list, f)

    process(config.train_source_data_path, config.train_data_path)
    process(config.valid_source_data_path, config.valid_data_path)
    print("数据处理完成！")
# from config import Config
# data_preprocess(Config())


class QATokensDataset(Dataset):
    def __init__(self, qa_tokens_list, qa_max_len):
        self.qa_tokens_list = qa_tokens_list
        self.qa_max_len = qa_max_len

    def __len__(self):
        return len(self.qa_tokens_list)

    def __getitem__(self, i):
        qa_tokens = self.qa_tokens_list[i][:self.qa_max_len] # 只截断，不填充
        return torch.tensor(qa_tokens, dtype=torch.long)


def load_dataset(config):
    with open(config.train_data_path, "rb") as f:
        train_input_list = pickle.load(f)
    with open(config.valid_data_path, "rb") as f:
        valid_input_list = pickle.load(f)

    return (
        QATokensDataset(train_input_list, config.qa_max_len),
        QATokensDataset(valid_input_list, config.qa_max_len),
    )
# from config import Config
# print(load_dataset(Config()))


# 填充
# 按照最长的句子来填充
def collate_fn(batch, config):
    return (
        rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0), # input tokens
        rnn_utils.pad_sequence(batch, batch_first=True, padding_value=config.padding_token), # label tokens
    )

def get_dataloader(config):
    train_dataset, valid_dataset = load_dataset(config)
    return (
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, config),
            drop_last=True,
        ),
        DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, config),
            drop_last=True,
        ),
    )
# from config import Config
# print(get_dataloader(Config()))