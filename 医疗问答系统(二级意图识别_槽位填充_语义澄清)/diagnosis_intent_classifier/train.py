import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from diagnosis_intent_classifier.data_utils import get_dataloader
from diagnosis_intent_classifier.model import DiagnosisIntentClassifierModel


def train(config):
	train_dataloader, _ = get_dataloader(config)
	model = DiagnosisIntentClassifierModel(config).to(config.device).train()

	my_optim = optim.Adam(model.parameters(), lr=config.diagnosis_intent_classifier_lr)
	loss_fn = nn.CrossEntropyLoss()

	total_sample_num = 0
	total_loss = 0
	total_accuracy = 0

	# 开始训练
	for epoch_i in range(1, config.diagnosis_intent_classifier_epochs+1):
		iterator = enumerate(tqdm(train_dataloader, desc=f'Diagnosis Intent 模型训练中... epoch: {epoch_i}'))
		for batch_i, (inputs, labels) in iterator:
			outputs = model(inputs)

			loss = loss_fn(outputs, labels)

			my_optim.zero_grad()
			loss.backward()
			my_optim.step()

			total_sample_num += outputs.size(0)
			total_loss += loss
			total_accuracy += sum(torch.argmax(outputs, dim=-1) == labels).item()

			if batch_i % 50 == 0:
				avg_loss = total_loss / total_sample_num
				avg_accuracy = total_accuracy / total_sample_num
				print(f'loss: {avg_loss:.4f}, accuracy: {avg_accuracy:.4f}')
		torch.save(model.state_dict(), config.diagnosis_intent_classifier_model_save_path) # 每个 epoch 保存一次

# from config import Config
# train(Config())