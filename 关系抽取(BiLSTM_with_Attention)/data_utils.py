from itertools import chain
from torch.utils.data import DataLoader, Dataset
import torch


# 填充和截断
def padding_and_cutoff(config, id_list, padding_value):
    padding_len = config.max_len-len(id_list)
    id_list.extend([padding_value]*padding_len) # 填充。即使 padding_len 是负数也没关系，只会生成 []
    return id_list[:config.max_len] # 截断


# word -> id, 同时填充和截断
def word_to_id_and_padding_cutoff(config, words):
    ids = [config.get_word_id(w) for w in words] # word -> id
    return padding_and_cutoff(config, ids, config.word_to_id['BLANK'])


# pos -> id, 同时填充和截断
def position_to_id_and_padding_cutoff(config, positions):
    ids = [pos + (config.max_len - 1) for pos in positions]
    # 共有 139 种可能的位置关系：-69 ~ 69 -> 0 ~ 138

    return padding_and_cutoff(config, ids, config.position_size - 1) # 最后增加一个位置 139 做填充, 共 140 种位置关系


# 读取一个数据文件，调整格式，增加位置信息
def get_data(config, data_path):
    relation_labels = [] # 给出的关系标签
    entities = [] # 给出的实体
    texts = [] # 原始文本
    positionE1 = [] # 原始文本相对于 E1 的位置关系
    positionE2 = [] # 原始文本相对于 E2 的位置关系
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            e1, e2, r, text = line.rstrip().split(' ', maxsplit=3) # 拆分一行: 实体1 实体2 关系 文本
            if r not in config.relation_to_id: continue # 忽略未知的关系

            # 计算位置信息
            e1_pos, e2_pos = text.index(e1), text.index(e2)
            position1, position2 = [], []
            for i in range(len(text)):
                position1.append(i-e1_pos)
                position2.append(i-e2_pos)

            # 存入各自的 list
            positionE1.append(position1)
            positionE2.append(position2)
            entities.append([e1, e2])
            relation_labels.append(config.relation_to_id[r])
            texts.append(list(text))

    return texts, relation_labels, positionE1, positionE2, entities


# 加载词表
# 在 config 创建时会自动调用
def load_vocab(config):
    texts = get_data(config, config.train_data_path)[0]

    # 把 texts 中的所有字去重后放到一个 list 中
    word_list = list(set(chain.from_iterable(texts))) # 共3866个不同的字
    word_list = sorted(word_list, key=ord) # 按 ASCII 码排序, 以保证每次构建的词表一致

    # 把 vocab 添加到 config 中
    config.word_to_id.update({vocab: idx for idx, vocab in enumerate(word_list, start=2)})
    config.id_to_word = {idx: vocab for vocab, idx in config.word_to_id.items()}

    config.vocab_size = len(config.word_to_id) # 共 3868 个字

    print('词表加载完成')


class MyDataset(Dataset):
    def __init__(self, config, data_path):
        self.data = get_data(config, data_path)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return [self.data[i][index] for i in (0,1,2,3,4)]


def collate_fn(batch, config):
    words_list = [data[0] for data in batch]
    labels = [data[1] for data in batch]
    positionE1 = [data[2] for data in batch]
    positionE2 = [data[3] for data in batch]
    entities = [data[4] for data in batch]

    # 处理 word
    word_ids_list = []
    for words in words_list:
        ids = word_to_id_and_padding_cutoff(config, words)
        word_ids_list.append(ids)

    # 处理 position
    positionE1_ids_list = []
    positionE2_ids_list = []
    for positions in positionE1:
        ids = position_to_id_and_padding_cutoff(config, positions)
        positionE1_ids_list.append(ids)
    for positions in positionE2:
        ids = position_to_id_and_padding_cutoff(config, positions)
        positionE2_ids_list.append(ids)

    # 都转为张量
    word_ids_list = torch.tensor(word_ids_list, dtype=torch.long)
    positionE1_ids_list = torch.tensor(positionE1_ids_list, dtype=torch.long)
    positionE2_ids_list = torch.tensor(positionE2_ids_list, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return word_ids_list, positionE1_ids_list, positionE2_ids_list, labels, words_list, entities


def get_dataloaders(config):
    train_data = MyDataset(config, config.train_data_path)
    test_data = MyDataset(config, config.test_data_path)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, config),
        drop_last=True
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, config),
        drop_last=True
    )

    return train_dataloader, test_dataloader




