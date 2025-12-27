import json
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
from random import choice

from config_and_model import CasRel


# 查找目标列表在源列表中的出现位置
# source_list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
#                        ⬆ ⬆ ⬆
# target_list: [4, 5, 6]
# 返回：3
def find_elements_pos(source_list, target_list):
	target_len = len(target_list)
	for i in range(len(source_list) - (target_len - 1)):
		if source_list[i:i + target_len] == target_list:
			return i
	return -1 # 没找到


class MyDataset(Dataset):
	def __init__(self, data_path):
		super(MyDataset, self).__init__()

		def process_line(line):
			sample = json.loads(line)
			return {
				'text': sample['text'],
				'triples': [(triple['subject'], triple['predicate'], triple['object']) for triple in sample['spo_list']]
				# triples: [(subject, relation, object)] - [('择天记', '作者', '猫腻')]
			}
		self.dataset = [process_line(line) for line in open(data_path, encoding='utf8')]

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, i):
		return self.dataset[i]
		# {
		# 	'text': '《姐妹们，上》，2015年台湾偶像剧，由姚采颖、张家慧、唐治平、黄采仪等领衔主演',
		# 	'triples': [('姐妹们，上', '主演', '张家慧'), ('姐妹们，上', '主演', '姚采颖'), ......]
		# }


# 这里每次只从 triple 中随机挑选 1 个主体及其对应的所有客体
def create_sample_inputs(triples, sentence_id_list, seq_len, config):
	# triples - [('偶尔寂寞', '作者', '阿琪'), ...]
	# sentence_id_list -  [101, 5356, 6782, 2972, ..., 0, 0, 0, 0, 0]
	# seq_len - 109

	sub__rel_obj_map = defaultdict(list)
	# {
	# 	(主体1头位置, 主体1尾位置): [(客体1头位置, 客体1尾位置, 关系), ...],
	# 	(主体2头位置, 主体2尾位置): [],
	# 	......
	# }

	# 把三元组加载到 sub__rel_obj_map 中
	for sub, rel, obj in triples:
		sub_ids = config.tokenizer(sub, add_special_tokens=False)['input_ids'] # (seq_len,)
		obj_ids = config.tokenizer(obj, add_special_tokens=False)['input_ids'] # (seq_len,)
		rel_id = config.relation_to_id[rel] # rel_id

		sub_head_pos = find_elements_pos(sentence_id_list, sub_ids) # pos_idx
		obj_head_pos = find_elements_pos(sentence_id_list, obj_ids) # pos_idx

		if sub_head_pos == -1 or obj_head_pos == -1: continue # 在文本中没找到三元组中的主体或客体

		sub_tail_pos = sub_head_pos + (len(sub_ids) - 1) # pos_idx
		obj_tail_pos = obj_head_pos + (len(obj_ids) - 1) # pos_idx

		sub__rel_obj_map[(sub_head_pos, sub_tail_pos)].append((obj_head_pos, obj_tail_pos, rel_id))

	subject_heads = torch.zeros(seq_len) # [0, 0, ..., 0]
	subject_tails = torch.zeros(seq_len) # [0, 0, ..., 0]
	object_heads = torch.zeros((seq_len, config.relation_num)) # [[0, 0, ..., 0], ..., [0, 0, ..., 0]]
	object_tails = torch.zeros((seq_len, config.relation_num)) # [[0, 0, ..., 0], ..., [0, 0, ..., 0]]
	subject_head2tail = torch.zeros(seq_len) # [0, 0, ..., 0]
	subject_len = torch.tensor([1]).float() # [1.] 用来做除法的分母，平均主语的权重，默认为 1，防止除 0 错误

	if sub__rel_obj_map: # 如果三元组不是空数组，才执行这里
		# 把[所有]出现的主体位置都记录到 subject_heads、subject_tails 中，用来训练模型从[文本中识别出实体]的能力
		for sub_head_pos, sub_tail_pos in sub__rel_obj_map.keys():
			subject_heads[sub_head_pos] = 1 # [1, 0, 0, 1, 0, 1, 0, ..., 0]
			subject_tails[sub_tail_pos] = 1 # [0, 1, 0, 1, 0, 0, 1, ..., 0]

		# 从三元组中只随机挑选一个主体及其对应所有客体，作为一个样本的 inputs，用来训练模型识别[客体+关系]的能力
		sub_head_pos, sub_tail_pos = choice(list(sub__rel_obj_map.keys()))
		subject_head2tail[sub_head_pos: sub_tail_pos + 1] = 1 # [0, 0, 0, 1, 1, 1, 0, 0, ..., 0]
		subject_len[0] = sub_tail_pos - sub_head_pos + 1 # [3.]
		for obj_head_pos, obj_tail_pos, rel_id in sub__rel_obj_map[(sub_head_pos, sub_tail_pos)]:
			object_heads[obj_head_pos, rel_id] = 1 # [[1, 0, ..., 0, 0], ..., [0, 0, ..., 1, 0]]
			object_tails[obj_tail_pos, rel_id] = 1 # [[0, 1, ..., 0, 0], ..., [0, 0, ..., 0, 1]]

	return {
		'subject_len': subject_len, # [num.]
		'subject_head2tail': subject_head2tail, # (seq_len)
		'subject_heads': subject_heads, # (seq_len)
		'subject_tails': subject_tails, # (seq_len)
		'object_heads': object_heads, # (seq_len, relation_num)
		'object_tails': object_tails, # (seq_len, relation_num)
	}


def collate_fn(batch, config):
	text_list = [sample['text'] for sample in batch] # 每个样本句子文本的 list. (batch_size, seq_len)
	tokenizer_output = config.tokenizer.batch_encode_plus(text_list, padding=True)
	# 这里 bert 会在首尾加入 [CLS]、[SEP]，并按照最长的句子进行填充
	sentence_id_list = [id_list for id_list in tokenizer_output['input_ids']] # 每个样本句子 id 的 list. (batch_size, seq_len)

	sample_inputs = [create_sample_inputs(
		batch[i]['triples'], # [(subject, relation, object), ...] - 文本
		sentence_id_list[i], # (seq_len,) - word id
		len(sentence_id_list[0]), # 填充后的句子长度 - num
		config
	) for i in range(config.batch_size)]

	tt = lambda x: torch.tensor(x).to(config.device)
	tt_stack = lambda x: torch.stack(x).to(config.device) # 类似于 extend()

	inputs = {
		'sentence_id_list': tt(sentence_id_list), # (batch_size, seq_len)
		'sentence_mask_list': tt([mask_list for mask_list in tokenizer_output['attention_mask']]), # 表示每个句子填充情况. (batch_size, seq_len)
		'subject_head2tail_list': tt_stack([inputs['subject_head2tail'] for inputs in sample_inputs]), # 主体的位置都是 1，其它位置是 0. (batch_size, seq_len)
		'subject_len_list': tt_stack([inputs['subject_len'] for inputs in sample_inputs]), # 主体长度的 list. (batch_size, 1)
	}
	labels = { # 形状都是：(batch_size, seq_len)
		'true_subject_heads': tt_stack([inputs['subject_heads'] for inputs in sample_inputs]),
		'true_subject_tails': tt_stack([inputs['subject_tails'] for inputs in sample_inputs]),
		'true_object_heads': tt_stack([inputs['object_heads'] for inputs in sample_inputs]),
		'true_object_tails': tt_stack([inputs['object_tails'] for inputs in sample_inputs]),
	}
	return inputs, labels


def get_data(config):
	train_dataloader = DataLoader(
		dataset=MyDataset(config.train_file),
		batch_size=config.batch_size,
		shuffle=True,
		collate_fn=lambda batch: collate_fn(batch, config),
		drop_last=True
	)
	dev_dataloader = DataLoader(
		dataset=MyDataset(config.dev_file),
		batch_size=config.batch_size,
		shuffle=False,
		collate_fn=lambda batch: collate_fn(batch, config),
		drop_last=True
	)
	test_dataloader = DataLoader(
		dataset=MyDataset(config.test_file),
		batch_size=config.batch_size,
		shuffle=False,
		collate_fn=lambda batch: collate_fn(batch, config),
		drop_last=True
	)
	return train_dataloader, dev_dataloader, test_dataloader


def extract_pos_pairs_from_0_1_list(predict_heads, predict_tails, config):
	# predict_heads - (seq_len) - [0, 0, 0, 1, 0, 1, 0, 0]
	# predict_tails - (seq_len) - [0, 0, 0, 0, 1, 1, 0, 0]

	# 找出所有值为1的元素索引位置
	pos_list= torch.arange(0, len(predict_heads)).to(config.device) # tensor([0, 1, 2, 3, 4, 5, 6, 7])
	head_pos_list = pos_list[predict_heads == 1].tolist() # [3, 5]
	tail_pos_list = pos_list[predict_tails == 1].tolist() # [4, 5]

	pos_pairs = zip(head_pos_list, tail_pos_list) # zip() - [(3, 4), (5, 5)]
	return [pair for pair in pos_pairs if pair[0] <= pair[1]] # [(head_pos, tail_pos), (head_pos, tail_pos), ...]


def extract_obj_and_rel(object_heads, object_tails, config):
	# object_heads/object_tails - (seq_len, relation_num) - [[0, 0, ..., 0, 1, 0, 0], [0, 0, ..., 0, 0, 1, 0], ...]

	# (seq_len, relation_num) -> (relation_num, seq_len)
	object_heads = object_heads.T
	object_tails = object_tails.T

	obj_and_rel_list = []
	for relation_id in range(config.relation_num):
		object_heads = object_heads[relation_id] # (seq_len)
		object_tails = object_tails[relation_id] # (seq_len)
		objects_pos = extract_pos_pairs_from_0_1_list(object_heads, object_tails, config) # [(start_pos, end_pos), ...]
		obj_and_rel_list.extend([(start_pos, end_pos, relation_id) for start_pos, end_pos in objects_pos])

	return obj_and_rel_list # [(start_pos, end_pos, relation_id), (start_pos, end_pos, relation_id), ...]


# torch.tensor([0.3, 0.7, 0.5, 0.9, 0.25]) -> tensor([0., 1., 1., 1., 0.])
convert_score_to_0_1 = lambda tensor: (tensor >= 0.5).float()


def load_model(config):
	model = CasRel(config).to(config.device).eval()
	model.load_state_dict(torch.load(config.model_save_path))
	return model