import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import os
import json
from sklearn.metrics import classification_report, confusion_matrix


# 对 BERT 进行微调
# 对用户的查询进行初步的意图分类：通用知识/专业咨询
# 注意：这里区分的是知识的类型，无法区分闲聊还是正经问题
class PrimaryQueryStrategySelector:
	def __init__(self, config):
		self.config = config
		self.tokenizer = BertTokenizer.from_pretrained(config.BERT_model_path)
		self.model = None # 需要使用 load_model 进行加载
		self.label_map = {"通用知识": 0, "专业咨询": 1} # 标签映射
		self._load_model()

	# 如果没有已训练好的模型，则加载预训练模型，否则加载已训练好的的模型
	def _load_model(self):
		final_model_save_path = self.config.intent_classification_final_model_save_path
		if os.path.exists(final_model_save_path / 'model.safetensors'): # 加载已经训练好的模型
			self.model = BertForSequenceClassification.from_pretrained(final_model_save_path)
		else: # 加载新的预训练模型进行训练
			self.model = BertForSequenceClassification.from_pretrained(self.config.BERT_model_path, num_labels=2)
		self.model.to(self.config.device)

	def _save_model(self):
		final_model_save_path = self.config.intent_classification_final_model_save_path
		self.model.save_pretrained(final_model_save_path)
		self.tokenizer.save_pretrained(final_model_save_path)

	# 用于交给 Trainer 做评估，在每个 epoch 结束时调用，打印结果
	@staticmethod
	def _compute_metrics(logits_list_and_labels):
		logits_list, labels = logits_list_and_labels
		predictions = np.argmax(logits_list, axis=-1)
		accuracy = (predictions == labels).mean()
		return {"accuracy": accuracy}

	# 训练结束时，评估模型并生成报告
	def _evaluate_model_and_gen_report(self, dataset: torch.utils.data.Dataset) -> None:
		trainer = Trainer(model=self.model)
		logits = trainer.predict(dataset).predictions
		predict_labels = np.argmax(logits, axis=-1)
		true_labels = [sample["labels"] for sample in dataset]

		with open(self.config.intent_classification_final_model_report_save_path, "w", encoding="utf-8") as f:
			f.write('分类报告：\n')
			f.write(classification_report(true_labels, predict_labels, digits=4, target_names=["通用知识", "专业咨询"]))
			f.write('\n\n混淆矩阵：\n')
			f.write(str(confusion_matrix(true_labels, predict_labels)))

	# 文本转为 token，label 转为数字, 作为 bert 的输入
	# 如果有 labels 也处理 labels
	def _tokenize(self, texts, labels=None):
		inputs = self.tokenizer(
			texts,
			truncation=True,
			padding='max_length',
			max_length=128,
			return_tensors="pt"
		)
		return inputs if not labels else (inputs, [self.label_map[label] for label in labels])

	def _create_dataset(self, inputs, labels):
		class Dataset(torch.utils.data.Dataset):
			def __init__(self, inputs, labels):
				super().__init__()
				self.inputs = inputs
				self.labels = labels

			def __len__(self):
				return len(self.labels)

			def __getitem__(self, i):
				dicts = {key: value[i] for key, value in self.inputs.items()}
				dicts["labels"] = torch.tensor(self.labels[i])
				return dicts
		return Dataset(inputs, labels)

	# 预测入口
	def intent_classification(self, query: str) -> str:
		inputs = self._tokenize([query])
		inputs = {k: v.to(self.config.device) for k, v in inputs.items()} # 转移到 GPU
		self.model.eval()
		with torch.no_grad():
			logits = self.model(**inputs).logits
		prediction = torch.argmax(logits, dim=1).item()
		return "专业咨询" if prediction == 1 else "通用知识"

	# 训练入口
	def train(self) -> None:
		# 读取数据
		with open(self.config.intent_classification_data_path, "r", encoding="utf-8") as f:
			data = [json.loads(line) for line in f.readlines()]
		texts = [sample["query"] for sample in data]
		labels = [sample["label"] for sample in data]

		# 划分数据集
		train_texts, test_texts, train_labels, test_labels = train_test_split(
			texts, labels,
			test_size=0.2, random_state=42, shuffle=True,
		)

		# tokenize
		train_inputs, train_labels = self._tokenize(train_texts, train_labels)
		test_inputs, test_labels = self._tokenize(test_texts, test_labels)

		# 创建数据集
		train_dataset = self._create_dataset(train_inputs, train_labels)
		test_dataset = self._create_dataset(test_inputs, test_labels)

		# 训练参数
		training_arguments = TrainingArguments(
			output_dir=self.config.intent_classification_checkpoint_model_save_path, # 检查点保存路径
			num_train_epochs=3, # epoch
			per_device_train_batch_size=8, # batch size
			per_device_eval_batch_size=8,  # batch size
			warmup_steps=20,  # 学习率预热的步数
			weight_decay=0.01,  # 权重衰减系数
			logging_steps=10,  # 每隔多少步打印日志
			eval_strategy="epoch",  # 每轮都进行评估
			save_strategy="epoch",  # 每轮都进行检查点的模型保存
			load_best_model_at_end=True,  # 加载最优的模型
			save_total_limit=1,  # 只保存一个检查点，其他被覆盖
			metric_for_best_model="eval_loss",  # 评估最优模型的指标（验证集损失）
			fp16=False,  # 禁用混合精度
		)

		# 训练器
		trainer = Trainer(
			model=self.model,
			args=training_arguments,
			train_dataset=train_dataset,
			eval_dataset=test_dataset,
			compute_metrics=self._compute_metrics, # 用于在每个 epoch 结束时计算评估指标并打印
		)

		trainer.train() # 开始训练

		self._save_model() # 保存模型

		self._evaluate_model_and_gen_report(test_dataset) # 最终评估


# from config import Config
# c = PrimaryQueryStrategySelector(Config)
# c.train()
# print(c.primary_query_strategy_selector("python课程的老师答疑及时吗？"))