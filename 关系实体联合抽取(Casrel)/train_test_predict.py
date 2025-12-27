import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm
from config_and_model import Config, CasRel
from utils import get_data, convert_score_to_0_1 as to_0_1, extract_pos_pairs_from_0_1_list, extract_obj_and_rel, load_model


pd.set_option('display.float_format', '{:.3f}'.format) # 打印 dataframe 时 float 数值只显示4位小数


# 用来在训练和最终测试中评估模型性能
def evaluate(model, dataloader, config):
	model.eval()

	total_loss = 0 # 累计本轮评估的总损失, 用来决定是否保存模型
	result = pd.DataFrame(
		columns=['Real', 'Predict', 'Correct', 'Precision', 'Recall', 'F1'], # 真实的对象个数/预测的对象个数/正确的对象个数/准确率/召回率/F1
		index=['Subject', 'Object_and_Relation'], # 要预测的两个部分
		dtype=float
	).fillna(0)
	rl = result.loc
	#                      Real  Predict  Correct  Precision  Recall  F1
	# Subject                 0        0        0          0       0   0
	# Object_and_Relation     0        0        0          0       0   0

	for inputs, labels in tqdm(dataloader, desc='评估中...'):
		outputs = model(**inputs)
		total_loss += model.loss_fn(**outputs, **labels)

		predict_subject_heads = to_0_1(outputs['predict_subject_heads']) # (batch_size, seq_len)
		predict_subject_tails = to_0_1(outputs['predict_subject_tails']) # (batch_size, seq_len)
		predict_object_heads = to_0_1(outputs['predict_object_heads']) # (batch_size, seq_len, relation_num)
		predict_object_tails = to_0_1(outputs['predict_object_tails']) # (batch_size, seq_len, relation_num)

		for i in range(config.batch_size): # 逐个样本处理
			# subjects 比对
			predict_subjects = extract_pos_pairs_from_0_1_list(predict_subject_heads[i], predict_subject_tails[i], config)
			true_subjects = extract_pos_pairs_from_0_1_list(labels['true_subject_heads'][i], labels['true_subject_tails'][i], config)
			# 格式：[(start_pos, end_pos), ......]
			rl['Subject', 'Real'] += len(true_subjects)
			rl['Subject', 'Predict'] += len(predict_subjects)
			rl['Subject', 'Correct'] += sum(1 for true_subject in true_subjects if true_subject in predict_subjects) # 反过来也行


			# objects & relations 比对
			predict_objects = extract_obj_and_rel(predict_object_heads[i], predict_object_tails[i], config)
			true_objects = extract_obj_and_rel(labels['true_object_heads'][i], labels['true_object_tails'][i], config)
			# 格式：[(start_pos, end_pos, relation_id), ...]
			rl['Object_and_Relation', 'Real'] += len(true_objects)
			rl['Object_and_Relation', 'Predict'] += len(predict_objects)
			rl['Object_and_Relation', 'Correct'] += sum(1 for true_object in true_objects if true_object in predict_objects)

	eps = 1e-10 # 防止除零
	rl['Subject', 'Precision'] = s_p = rl['Subject', 'Correct'] / (rl['Subject', 'Predict'] + eps)
	rl['Subject', 'Recall'] = s_r = rl['Subject', 'Correct'] / (rl['Subject', 'Real'] + eps)
	rl['Subject', 'F1'] = (2 * s_p * s_r) / (s_p + s_r + eps)

	rl['Object_and_Relation', 'Precision'] = o_p = rl['Object_and_Relation', 'Correct'] / (rl['Object_and_Relation', 'Predict'] + eps)
	rl['Object_and_Relation', 'Recall'] = o_r = rl['Object_and_Relation', 'Correct'] / (rl['Object_and_Relation', 'Real'] + eps)
	rl['Object_and_Relation', 'F1'] = (2 * o_p * o_r) / (o_p + o_r + eps)

	return result, total_loss


def train():
	config = Config()

	model = CasRel(config).to(config.device)

	# 准备优化器
	param_optimizer = list(model.named_parameters())
	no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"] # 不需要做 weight decay 的参数
	optimizer = AdamW([
		{"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
		{"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	], lr=config.lr, eps=10e-8)

	train_dataloader, dev_dataloader, _ = get_data(config)

	best_loss = float('inf')
	for epoch_i in range(1, config.epochs+1):
		for batch_i, (inputs, labels) in enumerate(tqdm(train_dataloader, desc=f'训练中... epoch: {epoch_i}')):
			model.train()
			outputs = model(**inputs) # 这里的警告忽略
			loss = model.loss_fn(**outputs, **labels).item()

			model.zero_grad()
			loss.backward()
			optimizer.step()

			if batch_i % 20 == 0:
				result, total_loss = evaluate(model, dev_dataloader, config)
				print(f'\n当前模型在验证集上的效果为：\n{result}\n-> loss: {total_loss:.5f}')
				if total_loss < best_loss:
					best_loss = total_loss
					torch.save(model.state_dict(), config.model_save_path)
					print('提升了提升了...... 模型已保存')


def final_test():
	config = Config()
	model = load_model(config)

	_, _, test_dataloader = get_data(config)

	with torch.no_grad():
		result, _ = evaluate(model, test_dataloader, config)

	print(f'模型性能测试结果：\n{result}')


def predict(text):
	config = Config()
	model = load_model(config).eval()

	# 先用模型预测 subject

	# 准备模型输入
	tokenizer_outputs = config.tokenizer(text) # 这里会在首尾加入 [CLS]、[SEP]
	sentence_id_list = torch.tensor([tokenizer_outputs['input_ids']]).to(config.device) # (batch_size, seq_len)
	sentence_mask_list = torch.tensor([tokenizer_outputs['attention_mask']]).to(config.device) # (batch_size, seq_len)

	# 用模型预测 subject 位置, 得到 subject_len 和 subject_head2tail
	with torch.no_grad():
		word_embedding_output = model.word_embedding_layer(sentence_id_list, sentence_mask_list) # (batch_size, seq_len, hidden_size)
		subject_heads, subject_tails = model.predict_subjects_pos(word_embedding_output) # (batch_size, seq_len)

	# 处理模型输出
	subject_heads = to_0_1(subject_heads[0]) # [batch_size, seq_len] -> [seq_len]
	subject_tails = to_0_1(subject_tails[0]) # [batch_size, seq_len] -> [seq_len]

	subject_head2tail_list = torch.zeros(1, len(sentence_id_list[0])).float().to(config.device) # (1, seq_len)
	subject_len_list = torch.tensor([[1]]).float().to(config.device) # [[1]]

	subjects_pos_pairs = extract_pos_pairs_from_0_1_list(subject_heads, subject_tails, config) # [(start_pos, end_pos), ...]

	if not subjects_pos_pairs:
		print('未识别出 subject-relation-object')
		return

	subject_start_pos, subject_end_pos = subjects_pos_pairs[0] # 只取第一个 subject，所以后面只会有一个 subject
	subject_len_list[0][0] = subject_end_pos - subject_start_pos + 1 # [[subject_len]]
	subject_head2tail_list[0][subject_start_pos:subject_end_pos+1] = 1 # (1, seq_len)

	# 再用模型去预测 object
	inputs = {
		'sentence_id_list': sentence_id_list, # (batch_size-1, seq_len)
		'sentence_mask_list': sentence_mask_list, # (batch_size-1, seq_len)
		'subject_head2tail_list': subject_head2tail_list, # (batch_size-1, seq_len)
		'subject_len_list': subject_len_list, # (batch_size-1, 1)
	}
	with torch.no_grad():
		outputs = model(**inputs)

	# 处理模型输出
	object_heads = to_0_1(outputs['predict_object_heads'][0]) # [batch_size, seq_len] -> [seq_len]
	object_tails = to_0_1(outputs['predict_object_tails'][0]) # [batch_size, seq_len] -> [seq_len]
	objects_pos_pairs = extract_obj_and_rel(object_heads, object_tails, config) # [(start_pos, end_pos, rel), ...]

	if not objects_pos_pairs:
		print('未识别出 subject-relation-object')
		return

	# 把id转为文本 (因为 bert 在词嵌入时添加了 [CLS] 和 [SEP]，所以这里必须先将 id 转为文本，才能进行后续处理)
	id_list = sentence_id_list[0] # (seq_len,)
	word_list = config.tokenizer.convert_ids_to_tokens(id_list) # (seq_len,) 文本

	print(f'Text: {''.join(word_list[1:-1])}\n-----------------------') # 打印原始文本

	for obj_start_pos, obj_end_pos, relation_id in objects_pos_pairs:
		subject_text = ''.join(word_list[subject_start_pos:subject_end_pos+1]) # subject 文本
		object_text = ''.join(word_list[obj_start_pos:obj_end_pos+1]) # object 文本
		relation_text = config.id_to_relation[relation_id] # relation 文本

		# 如果遇到尾部填充的 [PAD], 不要这一组结果
		if ('[PAD]' in subject_text) or ('[PAD]' in object_text): continue

		# 打印识别出的 subject-relation-object
		print(f'Subject: {subject_text}\tRelation: {relation_text}\tObject: {object_text}\n-----------------------')


# train()
# final_test()
# predict('《人间》是王菲演唱歌曲')