import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
from bert.train_eval_test_predict import train, final_test, predict
from bert.bert_model_and_config import Model as TeacherModel, Config as TeacherConfig
from textcnn_model_and_config import Model as StudentModel, Config as StudentConfig


# 蒸馏损失函数
def distillation_loss_fn(student_outputs, teacher_outputs, labels):
	alpha = 0.8 # 控制软损失和硬损失之间的权重分配
	T = 2 # 温度参数，影响损失函数输出的平滑程度

	student_outputs = F.log_softmax(student_outputs / T, dim=1)
	teacher_outputs = F.softmax(teacher_outputs / T, dim=1)
	# 下面的 KL 散度损失函数要求：
	# student_outputs(软预测) 的输入格式为  log(概率值)  (log_softmax 输出)
	# teacher_outputs(软目标) 的输入格式为  概率值       (softmax 输出)

	soft_loss = nn.KLDivLoss(reduction='batchmean')(student_outputs, teacher_outputs)
	hard_loss = nn.CrossEntropyLoss()(student_outputs, labels)

	return       alpha  * soft_loss * math.sqrt(T) \
		+ (1.0 - alpha) * hard_loss


# 获得教师模型输出
def get_teacher_outputs(student_config):
	# 优先尝试从文件加载
	if os.path.exists(student_config.save_teacher_outputs_path):
		print('正在从文件中加载教师模型输出...')
		return torch.load(student_config.save_teacher_outputs_path)

	# 否则从训练集计算并保存、返回
	teacher_config = TeacherConfig()
	teacher_model = TeacherModel(teacher_config).to(teacher_config.device)
	teacher_model.load_state_dict(torch.load(teacher_config.save_model_path))
	teacher_model.eval()
	with torch.no_grad():
		train_dataset_iterator = tqdm(teacher_config.get_dataset_iterator('train'), desc='教师模型推理中...')
		outputs = torch.cat([teacher_model(data_batch) for data_batch, _ in train_dataset_iterator])
		outputs = outputs.split(student_config.batch_size, dim=0)
		# 按照 student_config.batch_size 分割成 batch
		# ([batch_size, class_num], ...)

		# 保存并返回
		torch.save(outputs, student_config.save_teacher_outputs_path)
		return outputs


if __name__ == '__main__':
	# 训练部分
	# student_config = StudentConfig()
	# student_model = StudentModel(student_config).to(student_config.device)
	# teacher_outputs = get_teacher_outputs(student_config)
	# loss_fn = lambda outputs, labels, batch_idx: distillation_loss_fn(outputs, teacher_outputs[batch_idx], labels)
	# train(student_model, student_config, loss_fn)


	# 测试部分
	# student_config = StudentConfig()
	# student_model = StudentModel(student_config).to(student_config.device)
	# student_model.load_state_dict(torch.load(student_config.save_model_path))
	# final_test(student_model, student_config)


	# 推理部分
	# student_config = StudentConfig()
	# student_model = StudentModel(student_config).to(student_config.device)
	# student_model.load_state_dict(torch.load(student_config.save_model_path))
	# print(predict(student_model, student_config, '在学校中，对于青少年关于教育有很多的教学方法中考高考上课书本'))
	# # education

	pass
