import json
import torch
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from bert.bert_model_and_config import PublicConfig
from textcnn.utils import build_vocab
from torch.utils.tensorboard import SummaryWriter


class Config(PublicConfig):
	def __init__(self, device=torch.device('mps')):
		self.model_name = 'textcnn'
		self._root_dir = Path(__file__).parent.parent # 项目根目录路径
		self._textcnn_dir = f'{self._root_dir}/{self.model_name}'
		self.save_model_path = f'{self._textcnn_dir}/models/{self.model_name}.ckpt'
		self._logger = SummaryWriter(f'{self._textcnn_dir}/logs')
		self.accuracy_logger = lambda acc, idx: self._logger.add_scalar('Accuracy', acc, idx)
		self.device = device
		self.dropout = 0.5
		self.epochs = 5
		self.batch_size = 128
		self.learning_rate = 1e-3
		self.embed_dim = 256
		self.kernel_sizes = (2, 3, 4) # 卷积核大小, 一次卷积几个字
		self.kernel_num = 512 # 不同卷积核的数量
		self.eval_loss_fn = nn.CrossEntropyLoss() # 不用于训练，只用于内部验证
		self.save_teacher_outputs_path = f'{self._textcnn_dir}/teacher_outputs.pt'
		self.need_mask = False # 在生成 inputs 的过程中，是否需要添加 mask

		self.vocab_path = f'{self._root_dir}/data/{self.model_name}_vocab.csv'
		self.vocab_size = 0 # 词表大小, 在下面调用 build_vocab 时自动赋值
		self.seq_max_len = 32 # 句子最大长度
		self.MAX_VOCAB_SIZE = 10000 # 词表大小
		self.min_word_freq = 1 # 词频阈值，低于该阈值的词会被过滤掉
		self.UNK, self.PAD, self.CLS = "[UNK]", "[PAD]", "[CLS]" # unknown、padding、bert中综合信息符号
		self.UNK_idx = lambda : self.vocab_size - 2
		self.PAD_idx = lambda : self.vocab_size - 1
		self.CLS_idx = lambda : self.vocab_size

		# 加载词表
		build_vocab(self) # 要用它统计出 config.vocab_size
		with open(self.vocab_path, 'r', encoding='utf-8') as f:
			self.vocab = json.load(f)
		self.tokenize = list
		self.convert_tokens_to_ids = lambda words: [self.vocab.get(word, self.UNK_idx()) for word in words]

		PublicConfig.__init__(self)


class Model(nn.Module):
	def __init__(self, config):
		super(Model, self).__init__()

		self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.PAD_idx())

		self.conv_layers = nn.ModuleList([ # 卷积层列表，包含不同卷积核大小的卷积层
			nn.Conv2d(
				in_channels=1, # 输入通道数. embedding 的词向量输出是单通道的
				out_channels=config.kernel_num, # 输出通道数 - 不同卷积核的个数
				kernel_size=(kernel_size, config.embed_dim) # 卷积核的长和宽。这里卷积核的宽度和词嵌入维数相同，所以卷积结果宽度为1
			) for kernel_size in config.kernel_sizes
		])
		self.pool = lambda x: F.max_pool1d(x, kernel_size=x.size(2)) # 池化层 - 池化窗口的大小为前面卷积结果的宽度, 所以池化结果宽度为1

		self.dropout = nn.Dropout(config.dropout)
		self.fully_connected = nn.Linear(config.kernel_num * len(config.kernel_sizes), config.class_num)

	def forward(self, inputs):
		# 词嵌入
		embed_x = self.embedding(inputs) # inputs - tokens

		# 用不同尺寸的卷积核进行卷积和池化, 然后将结果拼接起来
		embed_x = embed_x.unsqueeze(1) # 调整为卷积层格式 - (batch_size, 1, word_num, embed_dim)
		outputs = []
		for conv_layer in self.conv_layers:
			# 卷积
			x = conv_layer(embed_x) # (batch_size, kernel_num, width = word_num - kernel_size + 1, height = 1)
			x = F.relu(x) # 形状不变
			x = x.squeeze(3) # (batch_size, kernel_num, width)

			# 池化(只有一维)
			x = self.pool(x) # (batch_size, kernel_num, 1)
			x = x.squeeze(2) # (batch_size, kernel_num)

			outputs.append(x)
		x = torch.cat(outputs, 1) # 拼接不同尺寸卷积核的结果
		# (batch_size, kernel_num * len(kernel_sizes)) - 不同卷积核的个数 * 卷积核的尺寸个数

		x = self.dropout(x)
		return self.fully_connected(x)