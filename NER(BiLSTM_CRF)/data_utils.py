import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import os


# 用来把有标注的数据整理到 train.txt 文件
def data_preprocess(config):
	# 解析标注文件
	def parse_tag_file(label_file_path):
		tag_dict = {}
		for line in open(label_file_path, 'r', encoding='utf-8'):
			_, start, end, label = line.strip().split('\t') # '疼痛\t27\t28\t症状和体征' -> ['疼痛', '27', '28', '症状和体征']
			start, end, label = int(start), int(end), config.name_to_label_dict.get(label) # label 换为英文
			for i in range(start, end + 1):
				tag_dict[i] = f'{'B' if i == start else 'I'}-{label}'
		return tag_dict # {27: 'B-SIGNS', 28: 'I-SIGNS', ......}

	with open(config.train_data_path, 'w', encoding='utf-8') as train_data_file:
		for abs_pos, child_dirs, child_files in os.walk(config.origin_data_root):
			for file_name in child_files:
				if 'original' not in file_name: continue # 不是数据文件
				file_path = f'{abs_pos}/{file_name}'
				tag_file_path = f'{abs_pos}/{file_name.replace('.txtoriginal','')}'
				tags_dict = parse_tag_file(tag_file_path)
				with open(file_path, 'r', encoding='utf-8') as f:
					content = f.read().strip()
					for i, char in enumerate(content):
						tag_id = tags_dict.get(i, 'O')
						train_data_file.write(f'{char}\t{tag_id}\n')
						# 文件写入效果：
						# '右    B-BODY'
						# '髋    I-BODY'
						# '部    I-BODY'
						# '摔    O'
						# '伤    O'
						# ......
# data_preprocess(Config())


# 构造数据集
def load_data_and_vocab(config):
	samples = []
	words, tags = [], []
	vocab_list = ["PAD", 'UNK'] # 初始词表

	for line in open(config.train_data_path, 'r', encoding='utf-8'):
		# 预处理
		sample = line.rstrip().split('\t') # ['咳', 'B-SIGNS']
		if (not sample) or (len(sample) != 2): continue
		word, tag = sample

		# 将单词记录到词表
		if word not in vocab_list:
			vocab_list.append(word)

		# 向样本中添加单词和tag
		words.append(word)
		tags.append(tag)

		# train.txt 中是没有分段的，这里遇到特定标点符号则存为一个样本
		if word in ['。', '?', '!', '！', '？']:
			samples.append([words, tags])
			words, tags = [], []

	# 构建词表，并添加到 config
	config.word_to_id_dict = {word: i for i, word in enumerate(vocab_list)}
	# {'PAD': 0, 'UNK': 1, '1': 2, '、': 3, '起': 4, '病': 5, '情': 6, ......}
	config.vocab_size = len(vocab_list)

	return samples
	# samples = [
	# 	[ # 样本1
	# 		['1', '、', '起', '病', '情', '况', ......],
	# 		['O', 'O', 'O', 'O', 'O', 'O', ......],
	# 	],
	# ]


# 需要定义这个 Dataset，以供 DataLoader 读取
class NerDataset(Dataset):
	def __init__(self, samples):
		super().__init__()
		self.samples = samples

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, i):
		return self.samples[i]


def get_data(config):
	samples = load_data_and_vocab(config) # 共有 7836 条样本

	def collate_fn(batch):
		# 把一个 batch 中 sample 的 tokens 和 tag 单独放在一起
		# 注意：这里因为每个 sample 的长度不一样，所以还不能把 torch.tensor 套在最外面，来把 x 和 y 转为张量
		x = [torch.tensor([config.word_to_id_dict[word] for word in sample[0]]) for sample in batch]
		y = [torch.tensor([config.tag_to_id_dict[tag] for tag in sample[1]]) for sample in batch]

		x_tokens = pad_sequence(x, batch_first=True, padding_value=0) # 自动将 batch 的长度填充为最长的
		y_true_tags = pad_sequence(y, batch_first=True, padding_value=0) # labels 一般用 -100 填充，但是因为 tag2id.json 中没有 -100，所以这里用 0 填充
		mask = (x_tokens != 0).bool()

		return x_tokens, y_true_tags, mask

	train_dataloader = DataLoader(
		dataset=NerDataset(samples[:6200]),
		batch_size=config.batch_size,
		collate_fn=collate_fn,
		drop_last=True,
		shuffle=True,
	)
	dev_dataloader = DataLoader(
		dataset=NerDataset(samples[6200:]),
		batch_size=config.batch_size,
		collate_fn=collate_fn,
		drop_last=True,
	)
	return train_dataloader, dev_dataloader