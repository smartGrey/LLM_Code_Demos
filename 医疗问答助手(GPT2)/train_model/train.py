import torch
import torch.nn.functional as F
import transformers
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
from config import Config
from train_model.data_utils import get_dataloader


# 模拟 GPT-2 内部在传入 labels 参数时，是如何用 cross_entropy 计算损失的
# 这个函数用不到
def calculate_loss(logit, labels, pad_idx=-100):
	# logit 截取 [  :-1]
	# label 截取 [1 :  ]
	# 正好错开一位
	logit = logit[:, :-1, :].contiguous().view(-1, logit.size(-1)) # (batch_size * seq_len, vocab_size)
	labels = labels[:, 1:].contiguous().view(-1) # (batch_size * seq_len)
	return F.cross_entropy(logit, labels, ignore_index=pad_idx)


# 根据 gpt2 模型输出计算准确率
def calculate_accuracy(logit, labels, ignore_index=-100):
	# logit 截取 [  :-1]
	# label 截取 [1 :  ]
	# 正好错开一位
	logit = logit[:, :-1, :].contiguous().view(-1, logit.size(-1)) # (batch_size * seq_len, vocab_size) - 二维
	labels = labels[:, 1:].contiguous().view(-1) # (batch_size * seq_len) - 一维 - [101, 342, 543, ... , -100, -100](后面是填充)

	# 对于每个 token，得到 logit 中预测概率最大的索引
	max_logit_values, max_logit_indexes = logit.max(dim=-1) # max_logit_indexes: [43, 467, 123, ....]

	# 指示 labels 中哪些位置不是填充的
	not_pad_mask = labels.ne(ignore_index) # [True, True, True, ..., False, False]
	not_pad_num = not_pad_mask.sum().item()

	# 将 logit 中预测概率最大的索引与标签张量进行比较，得到预测结果
	compared_result_idx = labels.eq(max_logit_indexes) # [True, True, False, ....] 一致的位置为 True
	compared_result_idx = compared_result_idx.masked_select(not_pad_mask) # 删除掉填充的部分, 得到一个更短的 bool 序列
	correct_num = compared_result_idx.sum().item() # 统计 True 的数量

	return correct_num, not_pad_num


def evaluate(model, config, valid_dataloader):
	total_loss = 0 # 计算每个 batch 的平均损失
	total_num, correct_num = 0, 0 # 计算准确率
	model.eval()
	with torch.no_grad():
		for input_token_ids, label_token_ids in tqdm(valid_dataloader, desc='验证集评估中...'):
			input_token_ids = input_token_ids.to(config.device)
			label_token_ids = label_token_ids.to(config.device)
			outputs = model(input_token_ids, labels=label_token_ids)
			# 如果这里输入了 labels，就可以直接获得 loss 值
			logits = outputs.logits
			loss = outputs.loss

			# 记录损失
			total_loss += loss.item()

			# 记录准确率
			batch_correct_num, batch_total_num = calculate_accuracy(logits, label_token_ids)
			total_num += batch_total_num
			correct_num += batch_correct_num
	average_loss = total_loss / len(valid_dataloader)
	accuracy = correct_num / total_num
	return average_loss, accuracy


def train(model, config):
	train_dataloader, valid_dataloader = get_dataloader(config)

	# 算出整个训练过程中，一共要更新多少次参数，也就是需要调整学习率的次数
	# 这里使用 get_linear_schedule_with_warmup 来动态调整学习率
	batch_num = len(train_dataloader)
	gradient_accumulation_step_num = config.gradient_accumulation_step_num
	epoch_num = config.epochs
	update_param_step_num = batch_num // gradient_accumulation_step_num * epoch_num # 训练过程中，一共要更新多少次参数

	# 参数优化器
	optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=config.eps)
	# 学习率动态调整器
	scheduler = transformers.get_linear_schedule_with_warmup( # 线性预热+线性衰减
		optimizer,
		num_warmup_steps=config.warm_up_proportion*update_param_step_num,
		num_training_steps=update_param_step_num,
	)

	best_valid_loss = 10000000
	for epoch_i in range(1, epoch_num+1):
		# 进行一个 epoch 的训练
		model.train()
		iterator = enumerate(tqdm(train_dataloader, desc=f'训练中...(epoch {epoch_i}/{epoch_num+1})'))
		for batch_i, (input_token_ids, label_token_ids) in iterator:
			input_token_ids = input_token_ids.to(config.device)
			label_token_ids = label_token_ids.to(config.device)
			outputs = model(input_token_ids, labels=label_token_ids) # 如果这里输入了 labels，就可以直接获得 loss 值

			# 对 loss 除以累积的步数
			loss = outputs.loss / gradient_accumulation_step_num # 每次只计算 1/num 的梯度
			loss.backward() # 反向传播

			# 梯度裁剪
			torch.nn.utils.clip_grad_norm_(
				model.parameters(), # 模型参数
				max_norm=config.max_grad_norm, # L2 范数的缩放阈值。如果超过这个阈值，则会按比例进行缩小
				norm_type=2, # L2 范数
			)

			# 进行一定step的梯度累计之后，更新参数
			if batch_i % config.gradient_accumulation_step_num == 0:
				optimizer.step() # 更新参数
				scheduler.step() # 更新学习率
				optimizer.zero_grad() # 梯度清零

		# 在验证集验证并保存模型
		valid_loss, valid_accuracy = evaluate(model, config, valid_dataloader)
		print(f'验证集损失为：{valid_loss:.4f}, 验证集准确率为：{valid_accuracy:.4f}')
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			model.save_pretrained(config.model_bin_path)


def main():
	config = Config()

	model_config = GPT2Config.from_json_file(config.model_config_path)
	model = GPT2LMHeadModel(config=model_config).to(config.device)
	# 这里需要保证 tokenizer 和 model 的 vocab_size 一致

	# 计算模型参数总量
	total_parameters_num = sum([parameter.numel() for parameter in model.parameters()])
	# 96069888

	train(model, config)
main()