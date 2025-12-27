from pathlib import Path
from data_utils import load_vocab
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
	def __init__(self):
		self.device = 'mps'
		root_dir = Path(__file__).parent
		self.train_data_path = f'{root_dir}/data/train.txt'
		self.test_data_path = f'{root_dir}/data/test.txt'
		self.save_model_path = f'{root_dir}/models/model.pth'

		self.max_len = 70 # 最长句子长度
		self.embedding_dim = 128 # 词嵌入维度
		self.position_dim = 32 # 用多少维来嵌入位置信息
		self.position_size = 2 * self.max_len # 一共可能产生多少种位置关系. 140: 0~139 + 1 (pos padding)
		self.lstm_hidden_output_dim = 256 # lstm 隐藏层输出总维度
		self.epochs = 500
		self.batch_size = 64
		self.lr = 1e-3
		self.stop_accuracy = 1

		self.relation_to_id = {
			'导演': 0,
			'歌手': 1,
			'作曲': 2,
			'作词': 3,
			'主演': 4,
		}
		self.id_to_relation = {idx: relation for relation, idx in self.relation_to_id.items()}
		self.relation_tag_size = len(self.relation_to_id)

		# 这些在调用 load_vocab 函数加载更新
		self.vocab_size = None
		self.word_to_id = {'UNKNOWN': 0, 'BLANK': 1}
		self.id_to_word = {idx: vocab for vocab, idx in self.word_to_id.items()}
		self.get_word_id = lambda word: self.word_to_id.get(word, self.word_to_id['UNKNOWN'])
		load_vocab(self)


class BiLSTM_Attention(nn.Module):
	def __init__(self, config):
		super(BiLSTM_Attention, self).__init__()

		c = self.config = config

		self.word_embedding = nn.Embedding(c.vocab_size, c.embedding_dim)
		self.position_embedding_1 = nn.Embedding(c.position_size, c.position_dim)
		self.position_embedding_2 = nn.Embedding(c.position_size, c.position_dim)
		self.lstm = nn.LSTM(
			input_size=c.embedding_dim + c.position_dim * 2,
			hidden_size=c.lstm_hidden_output_dim // 2,
			num_layers=1,
			bidirectional=True,
			batch_first=True
		)

		self.linear = nn.Linear(c.lstm_hidden_output_dim, c.relation_tag_size)

		self.dropout_embedding = nn.Dropout(p=0.2)
		self.dropout_lstm = nn.Dropout(p=0.2)
		self.dropout_attention = nn.Dropout(p=0.2)

		self.attention_weight = nn.Parameter(
			torch.randn(c.batch_size, 1, c.lstm_hidden_output_dim)
			.to(c.device)
		)

	def init_lstm_hidden(self):
		c = self.config
		h0 = torch.randn(2, c.batch_size, c.lstm_hidden_output_dim // 2).to(c.device)
		c0 = torch.randn(2, c.batch_size, c.lstm_hidden_output_dim // 2).to(c.device)
		return h0, c0

	def attention(self, H):
		M = F.tanh(H) # (batch_size, max_len, lstm_hidden_output_dim)

		# 注意力计算
		attention_weight = torch.bmm(self.attention_weight, M.transpose(1, 2))
		# self.attention_weight: (batch_size, 1, lstm_hidden_output_dim)
		# M.transpose_(1, 2): (batch_size, lstm_hidden_output_dim, max_len)
		# attention_weight: (batch_size, 1, max_len)

		a = F.softmax(attention_weight, dim=-1) # (batch_size, 1, max_len)

		# 应用注意力权重(加权平均池化)
		r = torch.bmm(a, H).squeeze(1)
		# a: (batch_size, 1, max_len)
		# H: (batch_size, max_len, lstm_hidden_output_dim)
		# r: (batch_size, 1, lstm_hidden_output_dim)
		# squeeze(1) -> (batch_size, lstm_hidden_output_dim)

		return F.tanh(r)

	def forward(self, words_list, positionE1_ids_list, positionE2_ids_list):
		x = torch.cat(( # (batch_size, max_len, embedding_dim + position_dim * 2)
			self.word_embedding(words_list), # (batch_size, max_len, embedding_dim)
			self.position_embedding_1(positionE1_ids_list), # (batch_size, max_len, position_dim)
			self.position_embedding_2(positionE2_ids_list) # (batch_size, max_len, position_dim)
		), -1)

		x = self.dropout_embedding(x)

		x, _ = self.lstm(x, self.init_lstm_hidden()) # (batch_size, max_len, lstm_hidden_output_dim)

		x = self.dropout_lstm(x)

		x = self.attention(x) # (batch_size, lstm_hidden_output_dim)

		x = self.dropout_attention(x)

		return self.linear(x) # (batch_size, relation_tag_size)
