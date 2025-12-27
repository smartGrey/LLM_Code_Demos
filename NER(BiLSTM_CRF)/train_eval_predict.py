import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from data_utils import get_data
from model_and_config import NER_LSTM_CRF, Config


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


def train():
	config = Config()
	train_dataloader, dev_dataloader = get_data(config)
	model = NER_LSTM_CRF(config).to(config.device)
	optimizer = optim.Adam(model.parameters(), lr=config.lr)

	best_f1_score = -1000
	for epoch_i in range(1, config.epochs+1):
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

			if batch_i % 200 == 0:
				f1_score, report = evaluate_model(model, dev_dataloader, config)
				print(f'f1_score: {f1_score:.3f}')
				if f1_score > best_f1_score:
					print(f'提升了提升了!')
					best_f1_score = f1_score
					torch.save(model.state_dict(), config.model_path)
					with open(config.report_path, "w") as f:
						print(report, file=f)


def ner(text):
	config = Config()

	# 处理模型输入
	x_tokens = torch.tensor([config.word_to_id(c) for c in text]).unsqueeze(0)
	mask = torch.ones(x_tokens.size(1)).bool().unsqueeze(0)

	# 准备模型对象
	model = NER_LSTM_CRF(config)
	model.load_state_dict(torch.load(config.model_path))
	model.eval()

	# 标注得到 tags
	tag_ids = model(x_tokens, mask)[0]
	# [0, 0, 3, 4, 4, ......]
	tags = [config.id_to_tag_dict[tag_id] for tag_id in tag_ids]
	# ['O', 'O', 'B-BODY', 'I-BODY', 'I-BODY', ......]

	# 对单个字的标注结果进行整理
	entities = []
	iterator = [(c, tag[0], config.label_to_name_dict.get(tag[2:], '')) for c, tag in zip(text, tags)]
	# '...常...' + [..., 'I-CHECK', ...] -> [('常', 'I', '检查和检验'), ...]
	for word, tag, label in iterator:
		if tag == 'B': # 实体开头
			entities.append((word, label)) # ('常', '检查和检验')
		elif tag == 'I' and entities: # 实体后部
			prev_word = entities[-1][0] # '血常'
			entities[-1] = (f'{prev_word}{word}', label) # ('血常规', '检查和检验')

	return entities

entities = ner('入院后完善各项检查，给予右下肢持续皮牵引，应用健骨药物治疗，患者略发热，查血常规：白细胞数12.18*10^9/L，中性粒细胞百分比92.00%。给予应用抗生素预防感染。复查血常规：白细胞数6.55*10^9/L，中性粒细胞百分比74.70%，红细胞数2.92*10^12/L，血红蛋白94.0g/L。考虑贫血，指示加强营养。建议患者手术治疗,患者拒绝手术治疗。继续右下肢牵引，患者家属要求今日出院。')
# [('右下肢', '身体部位'), ('健骨药物', '治疗'), ('发热', '症状和体征'), ('血常规', '检查和检验'), ('白细胞数', '检查和检验'), ('中性粒细胞百分比', '检查和检验'), ('抗生素', '治疗'), ('血常规', '检查和检验'), ('白细胞数', '检查和检验'), ('中性粒细胞百分比', '检查和检验'), ('红细胞数', '检查和检验'), ('血红蛋白', '检查和检验'), ('右下肢', '身体部位')]