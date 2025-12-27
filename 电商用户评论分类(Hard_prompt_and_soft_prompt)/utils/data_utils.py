import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator


# 输入：
#   texts: [
#   	'手机	这个手机也太卡了。',
#   	'体育	世界杯为何迟迟不见宣传',
#   	...
#   ]
#   data_type_processor: np.array / torch.LongTensor
# 输出：
#   {
#   	'input_ids_list': [[...], ...], - (batch, seq_len)
#   	'attention_mask_list': [[...], ...], - (batch, seq_len)
#   	'mask_positions_list': [[...], ...], - (batch, class_tokens_num)
#   	'mask_position_main_class_tokens': [(2372, 3442), (2643, 4434), ...] - (batch, class_tokens_num)
#       这里因为只有一句，所以 BERT 模型不传入 token_type_ids 这个字段也可以
#   }
def convert_dataset(texts, config):
	result = {
		'input_ids_list': [],
		'attention_mask_list': [],
		'mask_positions_list': [],
		'mask_position_main_class_tokens': []
	}
	for line in texts:
		label, text = line.strip().split('\t', 1) # ['手机', '这个手机也太卡了。'] - 只切第一次

		# 从 config 根据不同的 prompt mode 获取 input_text
		input_text = config.generate_input_text_on_prompt_template(text)
		# '这是一条关于[MASK][MASK]的评论：这个手机也太卡了。' 或者
		# '[unused1][unused2]......[unused6][MASK][MASK][unused7]......[unused10]这个手机也太卡了。'

		# 使用 bert_tokenizer 对 input_text 进行编码
		encoded_output = config.tokenizer(  # 直接调用相当于 encode_plus
			text=input_text,
			truncation=True,
			max_length=config.input_max_seq_len,
			padding='max_length',
			add_special_tokens=True,  # 首尾加入[CLS]和[SEP]
		)

		input_ids = encoded_output['input_ids'] # tokens
		# [101, 6821, ..., 754, 103, 103, 4638, ..., 511, 102, 0, 0, 0, ......]

		attention_mask = encoded_output['attention_mask']
		# 1 代表有效位置，0 代表填充位置 - [1, 1, ..., 1, 0, 0, ..., 0]

		mask_positions = np.where(np.array(input_ids) == config.mask_token_id)[0].tolist(),
		# 标记为 [MASK] 要进行填空预测的位置 - [7, 8]
		# 也就是 input_ids 中 103 的位置

		label_tokens = tuple(config.tokenizer.encode(label, add_special_tokens=False))
		# 答案对应的 tokens - (2372, 3442)
		# 这里编码之后，token 的数量有可能超过 class_tokens_num，取决于 tokenizer 的 词表，这里不需要处理

		result['input_ids_list'].append(input_ids)
		result['attention_mask_list'].append(attention_mask)
		result['mask_positions_list'].append(mask_positions)
		result['mask_position_main_class_tokens'].append(label_tokens)

	return {k: torch.LongTensor(v) for k, v in result.items()}


def get_data(config):
	# 从文件中加载数据
	datasets = load_dataset(
		'text', # 数据格式(类型)
		data_files={
			'train': config.train_data_path,
			'dev': config.dev_data_path,
		},
	)
	# DatasetDict({
	#     train: Dataset({ features: ['text'], num_rows: 453 })
	#     dev: Dataset({ features: ['text'], num_rows: 590 })
	# })

	# 将 dataset 内容转换为模型输入的 inputs, labels
	process_dataset = lambda dataset: convert_dataset(dataset['text'], config, data_type_processor=np.array)
	datasets = datasets.map(process_dataset, batched=True) # 用这个 map 可以保持里面仍然是 Dataset
	# DatasetDict({
	#     train: Dataset({
	#         features: ['text', 'input_ids_list', 'attention_mask_list', 'mask_positions_list', 'mask_position_main_class_tokens'],
	#         num_rows: 453
	#     })
	#     dev: Dataset({
	#         features: ['text', 'input_ids_list', 'attention_mask_list', 'mask_positions_list', 'mask_position_main_class_tokens'],
	#         num_rows: 590
	#     })
	# })

	# 转换为 dataloader
	return (
		DataLoader(
			datasets['train'],
			shuffle=True,
			collate_fn=default_data_collator,
			batch_size=config.batch_size,
			# drop_last=True,
		),
		DataLoader(
			datasets['dev'],
			collate_fn=default_data_collator,
			batch_size=config.batch_size,
			# drop_last=True,
		),
	)
# from config import Config
# config = Config('hard')
# train_dataloader, dev_dataloader = get_data(config)
# # batch_size 个样本的模型输入