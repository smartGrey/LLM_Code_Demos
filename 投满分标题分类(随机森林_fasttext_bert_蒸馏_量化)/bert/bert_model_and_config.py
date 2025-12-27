import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
from bert.utils import DatasetIterator
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class PublicConfig(object):
	def __init__(self):
		self._root_dir = Path(__file__).parent.parent # 项目根目录路径
		self.class_list = [x.strip() for x in open(f'{self._root_dir}/data/class.txt').readlines()]
		self.class_num = len(self.class_list)
		self.get_dataset_iterator = lambda mod: DatasetIterator(mod, self)


class Config(PublicConfig):
	def __init__(self, device=torch.device('mps')):
		self.model_name = 'bert'
		self._root_dir = Path(__file__).parent.parent # 项目根目录路径
		self._bert_dir = f'{self._root_dir}/{self.model_name}'
		self.save_model_path = f'{self._bert_dir}/models/{self.model_name}.ckpt'
		self.save_quantify_model_path = f'{self._bert_dir}/models/{self.model_name}_quantify.ckpt'
		self._logger = SummaryWriter(f'{self._bert_dir}/logs')
		self.accuracy_logger = lambda acc, idx: self._logger.add_scalar('Accuracy', acc, idx)
		self.class_list = [x.strip() for x in open(f'{self._root_dir}/data/class.txt').readlines()]
		self.class_num = len(self.class_list)
		self.device = device
		self.epochs = 1
		self.seq_max_len = 32
		self.batch_size = 128
		self.learning_rate = 5e-5
		self.hidden_size = 768 # bert 的隐藏层输出维度
		self.bert_path = f'{self._bert_dir}/models/bert_pretrain/'
		self._tokenizer = BertTokenizer.from_pretrained(self.bert_path)
		self.tokenize = self._tokenizer.tokenize
		self.convert_tokens_to_ids = self._tokenizer.convert_tokens_to_ids
		self.config = BertConfig.from_pretrained(self.bert_path+'/bert_config.json')
		self.eval_loss_fn = nn.CrossEntropyLoss() # 不用于训练，只用于内部验证
		self.need_mask = True # 在生成 inputs 的过程中，是否需要添加 mask
		self.PAD_idx = lambda : 0

		PublicConfig.__init__(self)


class Model(nn.Module):
	def __init__(self, config):
		super(Model, self).__init__()
		self.bert = BertModel.from_pretrained(config.bert_path, config=config.config)
		self.fully_connected = nn.Linear(config.hidden_size, config.class_num)

	def forward(self, inputs): # batch - (tokens, masks)
		tokens, masks = inputs

		_, pooled_output = self.bert(tokens, attention_mask=masks, return_dict=False)
		# pooled_output - 池化后的结果，将整个句子的信息压缩成一个固定长度的向量 - [batch_size, hidden_size]

		return self.fully_connected(pooled_output)