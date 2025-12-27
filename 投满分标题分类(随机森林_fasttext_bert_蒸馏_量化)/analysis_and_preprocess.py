import pandas as pd
from collections import Counter
import numpy as np
import jieba
from matplotlib import pyplot as plt


def data_look_over(name):
	df = pd.read_csv(f'./data/{name}.txt', sep='\t')

	df.head()
	# text                                             value
	# 中华女子学院：本科层次仅1专业招男生                     3
	# 东5环海棠公社230-290平2居准现房98折优惠                 1
	# 82岁老太为学生做饭扫地44年获授港大荣誉院士                5

	# 数据集杨恩数量
	len(df) # 180000

	# 样本类别分布
	Counter(df.label.values)
	# 3 18000
	# 4 18000
	# ......
	# 0 18000
	df.label.value_counts().plot.bar()
	plt.title('Labels Count')
	plt.xlabel('Label')
	plt.ylabel('Count')
	plt.show()

	# 样本句子长度分布
	df['len'] = df.text.apply(len) # 给 df 加一列 len
	axes = df.len.hist(bins=30)
	axes.set_title("Length Distribution")
	axes.set_xlabel("Length")
	axes.set_ylabel("Frequency")
	plt.show()
# data_look_over('train')
# data_look_over('test')


def read_data_and_process(path):
	# 读取
	df = pd.read_csv(path, sep='\t')

	# 用 jieba 分词 -> 最后将词语用空格连接为新的句子 -> 截断文本
	df['words'] = df.text.apply(lambda text: ' '.join(jieba.lcut(text.strip())))
	# 这里是按词进行分词的，也可以尝试按字切分，或许可以提高正确率

	# 截断文本
	words_length = df.words.apply(len)
	cut_off_len = np.ceil(words_length.mean() + words_length.std() * 3).astype(int) # 用 3σ 法则计算截断长度 - 47
	df['words'] = df.words.apply(lambda words: words[:cut_off_len].strip())

	return df


def data_process_for_random_forest(name):
	# 读取、处理、保存
	read_data_and_process(name)\
		.to_csv(f'./data/{name}_for_random_forest.csv', sep='\t', index=False)
# data_process_for_random_forest('./data/train.txt')
# data_process_for_random_forest('./data/test.txt')


def data_process_for_fast_text(name):
	# 读取、处理、转换格式、保存
	read_data_and_process(name)\
		.apply(lambda row: f'__label__{row["label"]} {row["words"]}', axis=1)\
		.to_csv(f'./data/{name}_for_fast_text.txt', index=False, header=None, sep='\t')
# data_process_for_fast_text('./data/train.txt')
# data_process_for_fast_text('./data/test.txt')