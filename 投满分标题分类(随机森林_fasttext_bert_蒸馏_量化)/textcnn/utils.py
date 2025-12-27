import os
import pandas as pd
from tqdm import tqdm
from bert.utils import padding_and_cutoff
import json


# 以 train 数据集构建词表
def build_vocab(config):
	if os.path.exists(config.vocab_path):
		print('词表已存在，正在读取词表大小...')
		with open(config.vocab_path, 'r', encoding='utf-8') as f:
			config.vocab_size = len(json.load(f))
		return

	print('正在构建词表...')

	# 统计词频
	vocab_count = {}
	df = pd.read_csv(f'../data/train.txt', sep='\t')
	texts = tqdm(df.text, desc='统计词频中...')
	for text in tqdm(df.text, desc='统计词频中...'): # 遍历每一行
		for word in text: # 遍历每个字
			vocab_count[word] = vocab_count.get(word, 0) + 1

	# 根据词频排序并筛选
	pairs = [(word, count) for word, count in vocab_count.items() if count >= config.min_word_freq] # [(word, count), ...] 词频太低就去掉
	vocab_list = sorted(pairs, key=lambda pair: pair[1], reverse=True)[:config.MAX_VOCAB_SIZE - 3] # 格式同上，按 count 降序排序，再取前 max_size-3 个
	vocab_dict = {word: i for i, (word, count) in enumerate(vocab_list, start=1)} # {word: token, ...} 遍历序号 i 作为 token, 从1开始

	# 添加特殊符号
	config.vocab_size = len(vocab_dict) # 下面三个特殊符号的索引以 vocab_size 为基准设置
	vocab_dict.update({
		config.UNK: config.UNK_idx(),
		config.PAD: config.PAD_idx(),
		config.CLS: config.CLS_idx()
	})
	config.vocab_size = config.vocab_size + 3 # 重新计算 vocab_size

	with open(config.vocab_path, 'w', encoding='utf-8') as f:
		json.dump(vocab_dict, f, ensure_ascii=False)