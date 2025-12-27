import json
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 固定随机种子
seed = 123
random.seed(seed)
np.random.seed(seed)


def data_preprocess(config):
	with open(config.slot_filling_ner_original_data, 'r', encoding='utf8') as f:
		datas = json.load(f)
	with open(config.slot_filling_ner_train_data, 'w', encoding='utf8') as f:  # w 会先清空文件
		for sample in tqdm(datas, desc='数据预处理中...'):  # 遍历每行数据
			text = sample['originalText']
			tags = [0] * len(text)  # 创建一个长度相同的数组用来标注
			for entity in sample['entities']:
				if '疾病和诊断' == entity['label_type']:
					start, end = entity['start_pos'], entity['end_pos']
					tags[start] = 1  # 标注 B_disease
					tags[start + 1:end] = [2] * len(text[start + 1:end])  # 标注 I_disease
			for word, tag in zip(text, tags):
				f.write(f'{word}\t{tag}\n')
	# 文件写入效果：
	# 	确	0
	# 	诊	0
	# 	肺	1
	# 	结	2
	# 	核	2
	# 	，	0
	# 	吃	0


# from config import Config
# data_preprocess(Config())


# 构造数据集和词表
def load_data_and_vocab(config):
	samples = []
	words, tags = [], []
	vocab_list = ["PAD", 'UNK']  # 初始词表

	for line in open(config.slot_filling_ner_train_data, 'r', encoding='utf-8'):  # 按行读入
		# 预处理
		sample = line.rstrip().split('\t')  # ['咳', 0]
		if (not sample) or (len(sample) != 2): continue
		word, tag = sample

		# 将单词记录到词表
		if word not in vocab_list:
			vocab_list.append(word)

		# 向样本中添加单词和tag
		words.append(word)
		tags.append(int(tag))

		# train.txt 中是没有分段的，这里遇到特定标点符号则存为一个样本
		if word in ['。', '?', '!', '！', '？']:
			samples.append([words, tags])
			words, tags = [], []

	# 构建词表，并添加到 config
	config.slot_filling_ner_word_to_id = {word: i for i, word in enumerate(vocab_list)}
	# {'PAD': 0, 'UNK': 1, '1': 2, '、': 3, '起': 4, '病': 5, '情': 6, ......}
	config.slot_filling_ner_vocab_size = len(vocab_list)

	return samples


# samples = [
# 	[ # 样本1
# 		['1', '、', '起', '病', '情', '况', ......],
# 		[0, 0, 0, 0, 0, 0, ......],
# 	],
# ]
# from config import Config
# print(load_data_and_vocab(Config())[0])


class SlotFillingNerDataset(Dataset):
	def __init__(self, samples):
		super().__init__()
		self.samples = samples

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, i):
		return self.samples[i]


def get_data(config):
	samples = load_data_and_vocab(config)  # 共有 7836 条样本
	random.shuffle(samples)  # 打乱数据

	def collate_fn(batch):
		# 把一个 batch 中 sample 的 tokens 和 tag 单独放在一起
		# 注意：这里因为每个 sample 的长度不一样，所以还不能把 torch.tensor 套在最外面，来把 x 和 y 转为张量
		x = [torch.tensor([config.slot_filling_ner_get_word_id(word) for word in sample[0]]) for sample in batch]
		y = [torch.tensor(sample[1]) for sample in batch]

		x_tokens = pad_sequence(x, batch_first=True, padding_value=0)  # 自动将 batch 的长度填充为最长的
		y_true_tags = pad_sequence(y, batch_first=True, padding_value=0)  # labels 一般用 -100 填充，但是因为 tag2id.json 中没有 -100，所以这里用 0 填充
		mask = (x_tokens != 0).bool()

		return x_tokens, y_true_tags, mask

	split_point = int(len(samples) * 0.85)
	train_dataloader = DataLoader(
		dataset=SlotFillingNerDataset(samples[:split_point]),
		batch_size=config.slot_filling_ner_batch_size,
		collate_fn=collate_fn,
		drop_last=True,
		shuffle=True,
	)
	dev_dataloader = DataLoader(
		dataset=SlotFillingNerDataset(samples[split_point:]),
		batch_size=config.slot_filling_ner_batch_size,
		collate_fn=collate_fn,
		drop_last=True,
		shuffle=False,
	)
	return train_dataloader, dev_dataloader