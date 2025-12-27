import numpy as np
import torch
import pandas as pd
from tqdm import tqdm


# predict 时使用它来处理输入
def convert_inputs_to_tensor(inputs, config):
	tt = lambda x: torch.tensor(x, dtype=torch.long).unsqueeze(0).to(config.device)
	# seq_max_len -> (batch_size, seq_max_len)

	if config.need_mask:
		tokens, mask = inputs
		return tt(tokens), tt(mask)
	else:
		return tt(inputs)


# 填充和截断 - (tokens, mask) / tokens
def padding_and_cutoff(tokens, max_len, pad_token, config):
	real_len = len(tokens)
	shortage = max_len - real_len # 不足的长度
	short = shortage>0 # 是否不足

	tokens = (tokens + [pad_token] * shortage) if short else tokens[:max_len]

	if config.need_mask:
		mask = ([1] * real_len + [0] * shortage) if short else ([1] * max_len)
		return tokens, mask
	else:
		return tokens


# 把文本处理为模型的输入 - (tokens, mask) / tokens
def text_to_model_input(text, config):
	words = (['[CLS]'] if config.model_name == 'bert' else []) + config.tokenize(text) # 分词：text -> word list
	tokens = config.convert_tokens_to_ids(words)  # word list -> token list
	return padding_and_cutoff(tokens, config.seq_max_len, config.PAD_idx(), config) # 填充和截断


# 取数据用的迭代器
class DatasetIterator(object):
	def __init__(self, mod, config):
		self.config = config

		df_values = pd.read_csv(f'../data/{mod}.txt', sep='\t').values
		self.dataset = [
			(text_to_model_input(text, self.config), label)
				for text, label in tqdm(df_values, desc=f'{mod} 数据集准备中...')
		] # 加载数据、分词、添加标记、填充和截断、创建mask - ((tokens, mask), label) / (tokens, label)

		# 如果是训练, 则打乱顺序 (这是一个原地操作)
		mod == 'train' and np.random.shuffle(self.dataset)

		# batch 相关
		self.batch_num = len(self.dataset) // self.config.batch_size # 有多少个 batch (不要最后不完整的 batch)
		self.next_batch_cursor = 0 # 下一次要返回的 batch 的索引

	# 调整 batch 的结构，并转为tensor
	# 把以 sample 切分的 batch 转为以 tokens、masks、labels 切分
	def _to_tensor(self, batch):
		if self.config.need_mask:
			tokens = torch.LongTensor([sample[0][0] for sample in batch]).to(self.config.device)
			masks = torch.LongTensor([sample[0][1] for sample in batch]).to(self.config.device)
			labels = torch.LongTensor([sample[1] for sample in batch]).to(self.config.device)
			return (tokens, masks), labels
		else:
			tokens = torch.LongTensor([sample[0].tolist()[0] for sample in batch]).to(self.config.device)
			labels = torch.LongTensor([sample[1] for sample in batch]).to(self.config.device)
			return tokens, labels

	def __next__(self): # 返回下一个 batch 的数据
		if self.next_batch_cursor >= self.batch_num:
			self.next_batch_cursor = 0
			raise StopIteration # 迭代结束, 不支持从头重新开始访问
		else:
			start_pos = self.next_batch_cursor * self.config.batch_size
			end_pos = start_pos + self.config.batch_size
			self.next_batch_cursor += 1
			return self._to_tensor(self.dataset[start_pos: end_pos])

	def __iter__(self):
		return self

	def __len__(self):
		return self.batch_num
# DatasetIterator('train', config)