import copy
import numpy as np
import peft
import torch
from peft import LoraConfig, TaskType
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, get_scheduler
from config import Config
from data_utils import get_data


def evaluate(model, test_dataloader, config):
	loss_list = []
	model.eval()
	with torch.no_grad():
		with torch.amp.autocast(config.device): # 混合精度训练
			for batch in tqdm(test_dataloader, desc='测试中...'):
				loss = model(
					input_ids=batch['input_ids_list'].to(config.device),
					labels=batch['label_ids_list'].to(config.device),
				).loss
				loss_list.append(loss.item())
	return sum(loss_list) / len(loss_list) # 返回平均loss


def train():
	config = Config()
	train_dataloader, test_dataloader = get_data(config)

	model = AutoModel.from_pretrained(
		config.pretrained_model_path,
		trust_remote_code=True,
		config=config.pretrained_model_config,
	)
	model.half().to(config.device) # 半精度加载到设备

	# 梯度检查点，降低显存占用
	# 不保存激活值，反向传播时重新计算
	model.gradient_checkpointing_enable()
	model.enable_input_require_grads()
	model.config.use_cache = False # 禁用缓存

	# 对于 p-tuning、输出头 部分的参数使用全精度
	model.transformer.prefix_encoder.float()
	model.lm_head.float()

	lora_config = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		inference_mode=False,
		r=config.lora_rank,
		lora_alpha=config.lora_alpha,
		lora_dropout=0.1,
	)
	model = peft.get_peft_model(model, lora_config)
	# model.print_trainable_parameters()
	# trainable params: 3,670,016 || all params: 6,222,831,616 || trainable%: 0.05897662393055503

	# 参数优化器
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		},
		{
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		},
	]
	optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)
	total_update_params_step_num = config.epochs * len(train_dataloader)
	lr_scheduler = get_scheduler(
		name='linear',
		optimizer=optimizer,
		num_warmup_steps=int(total_update_params_step_num * config.warmup_ratio),
		num_training_steps=total_update_params_step_num,
	)

	# 启用异常检测
	torch.autograd.detect_anomaly()

	best_evaluate_loss = float('inf')
	for epoch_i in range(1, config.epochs+1):
		for batch_i, batch in enumerate(tqdm(train_dataloader, desc=f'epoch {epoch_i} 训练中...')):
			model.train()
			with torch.amp.autocast(config.device):
				output = model(
					input_ids=batch['input_ids_list'].to(config.device, dtype=torch.long),
					labels=batch['label_ids_list'].to(config.device, dtype=torch.long),
				)
			loss = output.loss
			logits = output.logits

			# 比对预测结果
			# for i in range(len(batch['input_ids_list'])):
			# 	label_ids = batch['label_ids_list'][i] # [seq_len]
			# 	useful_pos = label_ids != -100
				# print('useful_pos -> ', useful_pos.shape, useful_pos)
				# label_ids = label_ids[useful_pos]


				# pred_ids = logits[i].argmax(dim=-1) # [seq_len, vocab_size] -> [seq_len]
				# print('pred_ids -> ', pred_ids.shape, pred_ids)
				# pred_ids = pred_ids[useful_pos]
				# print('pred_ids_after_choose -> ', pred_ids.shape, pred_ids)

				# print('='*20)
				# print('========== Sample ==========')
				# print(f'label len: {len(label_ids)} predict len: {len(pred_ids)}')
				# print('label   ids:', label_ids)
				# print('predict ids:', pred_ids)
				# print('========== inputs ==========')
				# print(config.tokenizer.decode(batch['input_ids_list'][i]))
				# print('========== label ==========')
				# print(config.tokenizer.decode(label_ids))
				# print('========== predict ==========')
				# print(config.tokenizer.decode(pred_ids))
				# print('='*20)
				# print()
				# print()



			optimizer.zero_grad()
			loss.backward()

			# 如果梯度范数过大(大于预设值0.5)，则进行梯度裁剪，防止梯度爆炸
			# print('max grad norm: ', max([p.grad.norm().item() for p in model.parameters() if p.grad is not None]))
			torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.grad_clip_norm)

			optimizer.step()
			lr_scheduler.step()

			print(f'[train batch loss]: {loss.item():.4f}')

			# if batch_i % config.evaluate_steps == 0:
			# 	evaluate_loss = evaluate(model, test_dataloader, config)
			# 	print(f'[evaluate loss]: {evaluate_loss:.4f}')
			# 	if evaluate_loss < best_evaluate_loss:
			# 		print('提升了提升了！ 模型保存中...')
			# 		best_evaluate_loss = evaluate_loss
			# 		copy.deepcopy(model).merge_and_unload().save_pretrained(config.best_model_save_path)
train()



def predict(model, config, instruction, input):
	input_text = f'Instruction: {instruction}\nInput: {input}\nAnswer: '
	input_ids = config.tokenizer.encode(input_text, return_tensors='pt').to(config.device)
	with torch.no_grad():
		output_ids = model.generate(
			input_ids=input_ids.to(config.device),
			max_new_tokens=config.max_output_len,
		)[0]
	return config.tokenizer.decode(output_ids).split('Answer: ')[-1]

def run_predict():
	config = Config()
	model = AutoModel.from_pretrained(
		config.best_model_save_path,
		trust_remote_code=True,
		config=config.pretrained_model_config,
	).half().to(config.device).eval()

	samples = [
		{
			'instruction': "现在你是一个非常厉害的SPO抽取器。",
			"input": "下面这句中包含了哪些三元组，用json列表的形式回答，不要输出除json外的其他答案。\n\n73获奖记录人物评价：黄磊是一个特别幸运的演员，拍第一部戏就碰到了导演陈凯歌，而且在他的下一部电影《夜半歌声》中演对手戏的张国荣、吴倩莲、黎明等都是著名的港台演员。",
		},
		{
			'instruction': "你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。",
			"input": "下面子中的主语是什么类别，输出成列表形式。\n\n第N次入住了，就是方便去客户那里哈哈。还有啥说的"
		}
	]
	for sample in samples:
		result = predict(model, config, sample['instruction'], sample['input'])
		print('='*20)
		print(result)
		print('='*20)
# run_predict()