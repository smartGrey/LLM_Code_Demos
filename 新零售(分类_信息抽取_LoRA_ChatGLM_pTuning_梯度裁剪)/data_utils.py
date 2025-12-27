import json
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator


# 从 train.jsonl 中计算出最合适的 max_input_len 和 max_output_len
def calc_best_cut_len(config):
	input_len_list = []
	output_len_list = []

	# 读取文件
	with open(config.train_data_path, 'r') as f:
		lines = f.readlines()

	# 记录输入和输出长度
	for l in lines:
		l = json.loads(l)
		input_len_list.append(len(config.tokenizer.encode(l['context'], add_special_tokens=False)))
		output_len_list.append(len(config.tokenizer.encode(l['target'], add_special_tokens=False)))

	# 计算最合适的 max_input_len 和 max_output
	get_2_sigma_upper = lambda num_list: np.mean(num_list) + np.std(num_list) * 2
	print('best input max len: ', get_2_sigma_upper(input_len_list), '. best output max len: ', get_2_sigma_upper(output_len_list)-3)
	# best input max len:  149.83100476867196 . best output max len:  156.34684443072135
# from config import Config
# config = Config()
# calc_best_cut_len(config)


# examples：['{'context': '...', 'target': '...'}', ...]
# 返回：
# 	(
# 		'input_ids_list': [[......], ...],
# 		'label_ids_list': [[......], ...],
# 		label_ids 与 input_ids 的位置一一对应，这里不需要错位，模型计算 loss 时会自动进行错位
#       长度全都是 max_input_and_output_len = max_input_len + max_output_len
# 	)
def convert_dataset(config, examples):
	result = {
		'input_ids_list': [],
		'label_ids_list': [],
	}
	tokenizer = config.tokenizer
	max_input_and_output_len = config.max_input_len + config.max_output_len

	for example in examples:
		example = json.loads(example) # str -> dict
		# {
		# 	'context': 'Instruction: 你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。\nInput: 句子中包含了哪些信息，输出json：\n\n如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈。\nAnswer: ',
		# 	'target': '```json\n[{"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "周星驰", "subject": "喜剧之王"}]\n```'
		# }

		# 编码为 token_id
		context_ids = tokenizer.encode(example['context'], add_special_tokens=False)
		target_ids = tokenizer.encode(example['target'], add_special_tokens=False)

		# 如果总长没超出限制，则不做处理
		# 如果超出限制，则直接对 context_ids 和 target_ids 进行截断
		# 保证最后 context_ids 和 target_ids 加起来不超过 max_input_and_output_len
		# (这里截断之后，长度总和很可能又不够了，因为很可能长出来的部分都在一个序列里，不过这里不做处理)
		if len(context_ids) + len(target_ids) > max_input_and_output_len - 3:
			context_ids = context_ids[:config.max_input_len]
			target_ids = target_ids[:config.max_output_len - 3] # 从 target_ids 里多留出 [gmask] [bos] [eos] 三个位置

		# 构建 input_ids
		input_ids = tokenizer.build_inputs_with_special_tokens(context_ids, target_ids)
		# 'context_ids [gmask] [bos] target_ids [eos]'
		# 总长 <= max_input_and_output_len

		# 构建 label_ids
		bos_pos = input_ids.index(tokenizer.bos_token_id)
		label_ids = [-100] * bos_pos + input_ids[bos_pos:]
		# [bos] 之前字符所对应的 label 全都是 -100, 模型遇到 -100 时会自动忽略损失计算
		# input_ids: 'context_ids [gmask] [bos] target_ids [eos]'
		# label_ids: '   -100      -100   [bos] target_ids [eos]'
		# 这里只是用字符来说明，实际上都是 id

		# 如果 context_ids + target_ids + 3 还是没超过 max_input_and_output_len
		# 则对 input_ids 和 label_ids 进行 padding
		# 注意 input_ids 用 pad_token_id 进行填充, label_ids 用 -100 进行填充
		pad_len = max_input_and_output_len - len(input_ids)
		input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
		label_ids = label_ids + [-100] * pad_len
		# input_ids: 'context_ids [gmask] [bos] target_ids [eos] [pad] [pad]'
		# label_ids: '   -100      -100   [bos] target_ids [eos] -100  -100'

		result['input_ids_list'].append(input_ids)
		result['label_ids_list'].append(label_ids)
	return result


def get_data(config):
	datasets = load_dataset(
		'text',
		data_files={
			'train': config.train_data_path,
			'test': config.test_data_path,
		},
	)
	datasets = datasets.map(
		lambda examples: convert_dataset(config, examples['text']),
		batched=True,
		load_from_cache_file=False,
	)
	return (
		DataLoader(
			datasets['train'],
			# shuffle=True,
			shuffle=False,
			collate_fn=default_data_collator,
			batch_size=config.batch_size,
		),
		DataLoader(
			datasets['test'],
			shuffle=False,
			collate_fn=default_data_collator,
			batch_size=config.batch_size,
		),
	)
# from config import Config
# config = Config()
# train_dataloader, test_dataloader = get_data(config)
# for batch in train_dataloader:
# 	input_ids = batch['input_ids_list'][0]
# 	label_ids = batch['label_ids_list'][0]
# 	print(input_ids)
# 	print(label_ids)
# 	print('-------')
# 	print(config.tokenizer.decode(input_ids))
# 	print('-------')
# 	print(config.tokenizer.decode(label_ids))
# 	print('-------')
# 	break
