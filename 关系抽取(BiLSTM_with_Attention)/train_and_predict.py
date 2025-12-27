import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_utils import get_dataloaders
from model_and_config import BiLSTM_Attention, Config
import numpy as np


def train(config):
	model = BiLSTM_Attention(config).to(config.device).train()
	optimizer = optim.Adam(model.parameters(), lr=config.lr)
	loss_fn = nn.CrossEntropyLoss()

	# 一些统计变量
	total_accuracy = 0  # 已经训练样本的准确率
	total_sample_num = 0 # 已经训练的样本数
	best_average_accuracy = 0 # 最佳平均准确率
	for epoch_i in range(config.epochs):
		train_iter = enumerate(tqdm(get_dataloaders(config)[0], desc=f'第 {epoch_i+1} 轮训练中...'))
		for batch_i, (words_list, positionE1_ids_list, positionE2_ids_list, labels, _, _) in train_iter:
			output = model(
				words_list.to(config.device), # (batch_size, max_len)
				positionE1_ids_list.to(config.device), # (batch_size, max_len)
				positionE2_ids_list.to(config.device) # (batch_size, max_len)
			).to('cpu')
			loss = loss_fn(output, labels)
			# output: (batch_size, relation_tag_size)
			# labels: (batch_size)
			# loss: tensor(1.5911, grad_fn=<NllLossBackward0>)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_accuracy += sum(torch.argmax(output, dim=1) == labels).item()
			total_sample_num += labels.size()[0]

			if batch_i % 50 == 0:
				average_accuracy = total_accuracy / total_sample_num
				if average_accuracy > best_average_accuracy:
					print('提升了提升了', end='')
					best_average_accuracy = average_accuracy
					torch.save(model.state_dict(), config.save_model_path)
				print(f'准确率：{average_accuracy:.4f}')
				if average_accuracy > config.stop_accuracy:
					return


def predict(config):
	model = BiLSTM_Attention(config).to(config.device).eval()
	model.load_state_dict(torch.load(config.save_model_path))

	predict_labels = []
	real_labels = []
	with torch.no_grad():
		for word_ids_list, positionE1_ids_list, positionE2_ids_list, true_labels, texts, entities_list in get_dataloaders(config)[1]:
			output = model(
				word_ids_list.to(config.device),
				positionE1_ids_list.to(config.device),
				positionE2_ids_list.to(config.device)
			).to('cpu')
			predict_labels.extend(torch.argmax(output, dim=1))
			real_labels.extend(true_labels)
	print('正确率：', np.sum(np.array(predict_labels) == np.array(real_labels)) / len(predict_labels))


# config = Config()
# train(config)
# predict(config)