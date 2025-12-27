import numpy as np
import torch
from sklearn import metrics
import time
from bert.bert_model_and_config import Config, Model
from bert.utils import text_to_model_input, convert_inputs_to_tensor
from tqdm import tqdm
from icecream import ic


def evaluate(model, config, mod):
	print(f'开始在 {mod} 数据集上进行评估...')

	total_loss = 0
	all_predicts = np.array([], dtype=int)
	all_labels = np.array([], dtype=int)

	model.eval()
	data_iterator = config.get_dataset_iterator(mod)

	with torch.no_grad():
		for inputs, labels in tqdm(data_iterator, desc=f'{mod} 数据集评估中...'):
			outputs = model(inputs)
			loss = config.eval_loss_fn(outputs, labels)
			total_loss += loss.item()

			# 把 labels 和 predicts 转为 numpy 数组
			labels = labels.data.cpu().numpy()
			predicts = torch.max(outputs.data, 1)[1].data.cpu().numpy()

			# 记录
			all_predicts = np.append(all_predicts, predicts)
			all_labels = np.append(all_labels, labels)
	accuracy = metrics.accuracy_score(all_labels, all_predicts)
	# accuracy_score 的参数必须是同一个类型的数组，不能 list 套 tensor
	# 所以上面都转成 numpy 数组, 这样或许可以比都转成 list 快一点
	average_loss = total_loss / len(data_iterator)

	if mod == 'dev': # 如果是验证集
		return accuracy, average_loss
	elif mod == 'test': # 如果是测试集，计算分类报告和混淆矩阵
		report = metrics.classification_report(all_labels, all_predicts, target_names=config.class_list)
		confusion = metrics.confusion_matrix(all_labels, all_predicts)
		return accuracy, average_loss, report, confusion


def train(model, config, loss_fn):
	print(f'{config.model_name} 模型开始训练...')
	best_dev_loss = float('inf') # 截止目前达到的最小损失

	# 参数分组优化(衰减)器
	param_optimizer = list(model.named_parameters()) # [('layer1.weight', tensor(...)), ('layer1.bias', tensor(...)), ...]
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	# 不需要衰减的参数 （这些参数通常对模型性能影响较小）
	# 对 bias 和 LayerNorm 参数不使用权重衰减，因为它们通常与模型的归一化或偏移相关，约束过强可能损害模型表达能力
	optimizer = torch.optim.AdamW([ # 其它的参数需要做衰减，防止过拟合
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} # 不衰减
	], lr=config.learning_rate) # 这里使用 L2 正则化
	# 相比传统 Adam，AdamW 将权重衰减与梯度更新解耦，使得衰减系数更稳定

	for epoch in range(1, config.epochs+1):
		iterator = tqdm(config.get_dataset_iterator('train'), desc=f'Epoch {epoch} Training...')
		for batch_idx, (inputs, labels) in enumerate(iterator, start=1): # inputs - (tokens, masks)
			model.train()
			outputs = model(inputs)
			loss = loss_fn(outputs, labels, batch_idx - 1) # 因为 batch_idx 从 1 开始
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch_idx % 20 == 0: # 每 100 个 batch 打印一次
				# 用 cpu 计算准确率
				labels = labels.data.cpu()
				predicts = torch.max(outputs.data, 1)[1].cpu()
				acc = metrics.accuracy_score(labels, predicts) # 本 batch 的准确率

				# 在测试集上验证
				model.eval()
				dev_acc, dev_loss = evaluate(model, config, 'dev')
				config.accuracy_logger(dev_acc, batch_idx) # 准确率打点用于 tensorboard 中查看

				# 如果损失小于阈值，则保存模型
				if dev_loss < best_dev_loss:
					torch.save(model.state_dict(), config.save_model_path)
					best_dev_loss = dev_loss
					print('提升了提升了，新模型已保存.')

				print(f'训练集 Acc: {acc:>.4f} 验证集 Acc: {dev_acc:>.4f} 验证集 Loss: {dev_loss:>.4f}')
	print('训练完成.')


def final_test(model, config):
	print('正在做最终测试......')

	start_time = time.time()
	test_accuracy, test_average_loss, report, confusion = evaluate(model, config, 'test')

	ic(test_accuracy)
	ic(test_average_loss)
	ic(report)
	ic(confusion)

	print('测试完成.')


def predict(model, config, text):
	model.eval()
	inputs = text_to_model_input(text, config) # (tokens, masks)
	inputs = convert_inputs_to_tensor(inputs, config)
	outputs = model(inputs) # (batch_size, class_num)
	return config.class_list[torch.max(outputs.data, 1)[1]] # 返回预测的标签


if __name__ == '__main__':
	# 训练部分
	# config = Config()
	# model = Model(config).to(config.device)
	# loss_fn = lambda outputs, labels, batch_idx: config.eval_loss_fn(outputs, labels)
	# train(model, config, loss_fn)

	# 测试部分
	# config = Config()
	# model = Model(config).to(Config().device)
	# model.load_state_dict(torch.load(Config().save_model_path))
	# final_test(model, config)

	# 推理部分
	# config = Config()
	# model = Model(config).to(Config().device)
	# model.load_state_dict(torch.load(Config().save_model_path))
	# print(predict(model, config, '日本地震：金吉列关注在日学子系列报道'))
	# education

	pass
