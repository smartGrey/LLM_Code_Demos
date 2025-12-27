from pathlib import Path
import torch
from torch import nn
from transformers import BertTokenizer, BertModel


class Config(object):
	def __init__(self):
		self.device = 'mps'

		self.bert_name = 'google-bert/bert-base-chinese'
		self.tokenizer = BertTokenizer.from_pretrained(self.bert_name) # 需要联外网

		root_dir = Path(__file__).parent
		self.train_file = f'{root_dir}/data/train.json'
		self.test_file = f'{root_dir}/data/test.json'
		self.dev_file = f'{root_dir}/data/dev.json'
		self.model_save_path = f'{root_dir}/models/model.pth'

		self.batch_size = 32
		self.lr = 1e-5
		self.bert_dim = 768
		self.epochs = 10


		self.id_to_relation = [ # 数组的下标作为关系的 id
			'出品公司', "国籍", "出生地", "民族", "出生日期", "毕业院校",
			"歌手", "所属专辑", "作词", "作曲", "连载网站", "作者",
			"出版社", "主演", "导演", "编剧", "上映时间", "成立日期",
		]
		self.relation_to_id = {relation: i for i, relation in enumerate(self.id_to_relation)}
		self.relation_num = len(self.id_to_relation) # 18


class CasRel(nn.Module):
	def __init__(self, config):
		super().__init__()
		c = self.config = config

		# 词嵌入层
		self._bert = BertModel.from_pretrained(c.bert_name) # 这个一定要在 __init__ 中创建好模型才能正确加载，不能套在下面函数中
		self.word_embedding_layer = lambda sentence_id_list, sentence_mask_list: \
			self._bert(sentence_id_list, attention_mask=sentence_mask_list)[0] # (batch_size, seq_len, hidden_size)

		# 预测 subject 的两个线性层
		self.sub_heads_linear = nn.Linear(in_features=c.bert_dim, out_features=1)
		self.sub_tails_linear = nn.Linear(in_features=c.bert_dim, out_features=1)

		# 预测 object 的两个线性层
		self.obj_heads_linear = nn.Linear(in_features=c.bert_dim, out_features=c.relation_num)
		self.obj_tails_linear = nn.Linear(in_features=c.bert_dim, out_features=c.relation_num)

	# 预测 subject 的位置
	def predict_subjects_pos(self, word_embedding_output):
		return (
			torch.sigmoid(self.sub_heads_linear(word_embedding_output)).squeeze(-1), # predict_subject_heads
			torch.sigmoid(self.sub_tails_linear(word_embedding_output)).squeeze(-1), # predict_subject_tails
			# (batch_size, seq_len)
		)

	# 预测 object 的位置
	def predict_objects_pos(
		self,
		subject_head2tail_list, # (batch_size, 1, seq_len)
		subject_lens, # (batch_size, 1)
		word_embedding_output # (batch_size, seq_len, hidden_size)
	):
		subject_embedding_output = torch.matmul(subject_head2tail_list, word_embedding_output) / subject_lens.unsqueeze(1)
		# 将[主实体特征]和[文本的词嵌入结果]进行融合:
		#       (batch_size, 1, seq_len) * (batch_size, seq_len, hidden_size) -> (batch_size, 1, hidden_size)
		# 对 subject_lens 进行升维，以进行计算. 计算时还自动对 subject_lens 进行了广播:
		#       (batch_size, 1, 1) -> (batch_size, 1, hidden_size)
		# 将嵌入 subject 特征后的词嵌入结果除以 subject 长度进行平均，防止 subject 长度不同影响嵌入

		# 与文本的原始编码进行融合，得到新的词嵌入结果
		new_embedding_output = word_embedding_output + subject_embedding_output # (batch_size, seq_len, hidden_size)

		# 用线性层进行预测
		objects_heads_list = torch.sigmoid(self.obj_heads_linear(new_embedding_output))
		objects_tails_list = torch.sigmoid(self.obj_tails_linear(new_embedding_output))
		# (batch_size, seq_len, relation_num)

		return objects_heads_list, objects_tails_list

	# 前向传播
	def forward(self, sentence_id_list, sentence_mask_list, subject_head2tail_list, subject_len_list):
		# sentence_id_list / sentence_mask_list / subject_head2tail - (batch_size, seq_len)
		# subject_len - (batch_size, 1)

		word_embedding_output = self.word_embedding_layer(sentence_id_list, sentence_mask_list)
		# (batch_size, seq_len, hidden_size)

		# 对 subject 位置进行预测
		predict_subject_heads, predict_subject_tails = self.predict_subjects_pos(word_embedding_output)
		# (batch_size, seq_len)

		objects_heads, objects_tails =self.predict_objects_pos(
			subject_head2tail_list.unsqueeze(1), # (batch_size, 1, seq_len)
			subject_len_list, # (batch_size, 1)
			word_embedding_output # (batch_size, seq_len, hidden_size)
		) # (batch_size, seq_len, relation_num)

		return {
			# subject 位置预测结果
			'predict_subject_heads': predict_subject_heads,
			'predict_subject_tails': predict_subject_tails,
			# 都是 (batch_size, seq_len)

			# object 位置预测结果
			'predict_object_heads': objects_heads,
			'predict_object_tails': objects_tails,
			# (batch_size, seq_len, relation_num)

			'mask': sentence_mask_list,
			# (batch_size, seq_len)
		}

	# 计算位置预测分数矩阵的损失
	@staticmethod
	def _calc_loss(predict_pos, target_pos, mask):
		loss = nn.BCELoss(reduction='none')(predict_pos, target_pos)
		# reduction='none': 不进行平均(降维)
		# sub - (batch_size, seq_len) / obj - (batch_size, seq_len, relation_num)

		return torch.sum(loss * mask) / torch.sum(mask) # 按照 mask 中的实际序列长度进行平均

	# 模型的损失函数
	def loss_fn(
		self,
		# 来自 model(forward) 的输出:
		predict_subject_heads, predict_subject_tails, predict_object_heads, predict_object_tails, mask,
		# 来自 collate_fn 提供的 labels:
		true_subject_heads, true_subject_tails, true_object_heads, true_object_tails
	):
		obj_mask = mask.unsqueeze(-1).repeat(1, 1, true_object_heads.shape[-1])
		# (batch_size, seq_len, 1) -> (batch_size, seq_len, relation_num)

		return (
			self._calc_loss(predict_subject_heads, true_subject_heads, mask)  # (batch_size, seq_len)
			+ self._calc_loss(predict_subject_tails, true_subject_tails, mask)  # (batch_size, seq_len)
			+ self._calc_loss(predict_object_heads, true_object_heads, obj_mask)  # (batch_size, seq_len, relation_num)
			+ self._calc_loss(predict_object_tails, true_object_tails, obj_mask) # (batch_size, seq_len, relation_num)
		)