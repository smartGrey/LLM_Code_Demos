import random

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from data_utils import get_data
from slot_filling_ner.model import SlotFillingNerModel

# 固定随机种子
seed = 123
random.seed(seed)
np.random.seed(seed)


def evaluate_model(model, dev_dataloader, config):
	model.eval()

	all_predicts, all_targets = [], [] # (all_samples_num * seq_len)
	# 因为 f1_score 没法接受多维的结果，所以这里不按 batch 分割, 也不按样本分割，直接把所有句子的 tokens 堆进同一个数组

	iterator = enumerate(tqdm(dev_dataloader, desc="测试集验证中...", mininterval=8)) # 8 秒才刷新一次
	for batch_i, (x_tokens_batch, y_true_tags_batch, mask_batch) in iterator:
		x_tokens_batch = x_tokens_batch.to(config.device) # (batch_size, seq_len)
		mask_batch = mask_batch.to(config.device)

		y_predict_tags_batch = model(x_tokens_batch, mask_batch)

		# 遍历 batch 中每个样本的 tokens，算出每个 x 序列的实际长度（即不是 PAD(0)）
		# 这里实际上是要得出 y 的有效长度，但是因为 y 和 x 的长度是一致的(因为每个token都有tag)，所以直接用 x 计算长度
		for sample_i, tokens in enumerate(x_tokens_batch.cpu()): # 复杂运算放到 CPU 上速度更快
			true_len = len(list(filter(lambda token: token!=0, tokens)))
			all_predicts.extend(y_predict_tags_batch[sample_i][:true_len]) # 这个本来就是 list。直接把元素插入数组而不是append
			all_targets.extend(y_true_tags_batch[sample_i][:true_len].tolist())
	return f1_score(all_predicts, all_targets, average='macro'), \
		   classification_report(all_predicts, all_targets)


def train(config, continue_train=False):
	train_dataloader, dev_dataloader = get_data(config)
	model = SlotFillingNerModel(
		config, config.slot_filling_ner_model_save_path if continue_train else None
	).to(config.device)
	optimizer = optim.Adam(model.parameters(), lr=config.slot_filling_ner_lr)

	best_f1_score = -1000
	for epoch_i in range(1, config.slot_filling_ner_epochs+1):
		model.train()
		iterator = enumerate(tqdm(train_dataloader, desc=f'epoch-{epoch_i} 训练中...'), start=1)
		for batch_i, (x_tokens, y_true_tags, mask) in iterator:
			x_tokens = x_tokens.to(config.device)
			y_true_tags = y_true_tags.to(config.device)
			mask = mask.to(config.device)

			loss = model.loss_fn__log_likelihood(x_tokens, y_true_tags, mask).mean() # 对损失求平均值

			optimizer.zero_grad()
			loss.backward()

			torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
			# 梯度裁剪, 防止梯度爆炸
			# 将梯度上限设为 max_norm，如果超过，会将所有梯度按比例缩小到该范围内

			optimizer.step()

			if batch_i % 150 == 0:
				f1_score, report = evaluate_model(model, dev_dataloader, config)
				print(f'f1_score: {f1_score:.3f} loss: {loss:.4f}')
				if f1_score > best_f1_score:
					print(f'提升了提升了!')
					best_f1_score = f1_score
					torch.save(model.state_dict(), config.slot_filling_ner_model_save_path)
				with open(config.slot_filling_ner_get_report_save_path(f'{epoch_i}-{batch_i}'), "w") as f:
					print(report, file=f)

# from config import Config
# train(Config(), continue_train=True)