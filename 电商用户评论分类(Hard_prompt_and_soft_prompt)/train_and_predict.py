import json
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, get_scheduler
from config import Config
from utils.common import Evaluator, get_predict_sub_class_ids, loss_fn
from utils.data_utils import get_data, convert_dataset


# 注意：这里的 train 和 predict 如果使用不同的 mode
# 那么训练、保存、使用的是不同的模型、prompt_template


def evaluate(model, eval_dataloader, config):
	id_tuple_to_word = lambda ids: ''.join(config.tokenizer.convert_ids_to_tokens(ids)) # (2372, 3442) -> '电器'
	evaluator = Evaluator(config)
	model.eval()
	with torch.no_grad():
		for batch in tqdm(eval_dataloader, desc='评估中...'):
			logits = model(
				input_ids=batch['input_ids_list'].to(config.device),
				attention_mask=batch['attention_mask_list'].to(config.device),
			).logits

			# 将预测分数转换为 sub_class_token_ids - [(2372, 3442), ...]
			predict_sub_class_ids = get_predict_sub_class_ids(logits, batch['mask_positions_list'], config)

			# 根据 sub_class_token_ids 找到 main_class_token_ids
			predict_main_class_ids = config.classMapper.batch_find_main_class_ids_by_inaccurate_sub_class_ids(predict_sub_class_ids)
			# [(2372, 3442), ...]

			# 添加到评估器中
			evaluator.add_batch_results_and_labels(
				[id_tuple_to_word(e) for e in batch['mask_position_main_class_tokens']], # 答案 - ['电器', ...]
				[id_tuple_to_word(e) for e in predict_main_class_ids], # 预测结果 - ['电器', ...]
			)
	return evaluator.compute_metrics() # 用评估器对比结果


def train(prompt_mode):
	config = Config(prompt_mode)
	model = AutoModelForMaskedLM.from_pretrained(config.pretrained_model_name).to(config.device)
	# 这里用的是带 MLM 任务头的，用于做 MASK 预测的模型
	# 这里有警告说部分参数未使用是正常的，因为没有使用完整的模型
	train_dataloader, dev_dataloader = get_data(config)

	# 设置优化器
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
	total_train_update_params_steps = config.epochs * len(train_dataloader) # 训练总步数 epochs * batch_size
	lr_scheduler = get_scheduler(
		name='linear', # 线性学习率预热
		optimizer=optimizer,
		num_warmup_steps=int(config.warm_up_ratio * total_train_update_params_steps),
		num_training_steps=total_train_update_params_steps,
	)

	best_f1 = 0
	batch_loss_list = []

	for epoch_i in range(1, config.epochs + 1):
		iterator = enumerate(tqdm(train_dataloader, desc=f'训练中... epoch: {epoch_i}'), start=1)
		for batch_i, batch in iterator:
			model.train()
			logits = model(
				input_ids=batch['input_ids_list'].to(config.device),
				attention_mask=batch['attention_mask_list'].to(config.device),
			).logits.cpu() # loss 需要在cpu上计算

			# 根据 true_main_class_ids 找到 true_sub_classes_ids_list，用于计算损失
			true_main_class_ids = batch['mask_position_main_class_tokens'] # 答案 - [(2372, 3442), ...]
			true_sub_classes_ids_list = config.classMapper.batch_find_sub_classes_ids_list_by_main_class_ids(true_main_class_ids)
			loss = loss_fn(logits, batch['mask_positions_list'], true_sub_classes_ids_list, config)
			batch_loss_list.append(loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			lr_scheduler.step()

			# 打印损失
			if batch_i % config.logging_steps == 0:
				print(f'loss: {sum(batch_loss_list) / len(batch_loss_list):.4f}')

		# 验证并保存模型
		report = evaluate(model, dev_dataloader, config)
		f1_score = report['f1']
		print(f'f1 score: {f1_score}')
		if f1_score > best_f1:
			best_f1 = f1_score
			model.save_pretrained(config.best_model_save_path)
			json.dump(report, open(config.best_model_report_save_path, 'w'), ensure_ascii=False)
			print('提升了提升了! 模型已保存。')
# train('hard')
# train('soft')


def predict(text_list, prompt_mode):
	config = Config(prompt_mode)
	model = AutoModelForMaskedLM.from_pretrained(config.best_model_save_path).to(config.device).eval()

	inputs = convert_dataset(map(lambda text: f'XX\t{text}', text_list), config) # 凑训练集的格式
	logits = model(
		inputs['input_ids_list'].to(config.device),
		inputs['attention_mask_list'].to(config.device),
	).logits

	# 将预测分数转换为 sub_class_token_ids - [(2372, 3442), ...]
	predict_sub_class_ids = get_predict_sub_class_ids(logits, inputs['mask_positions_list'], config)

	# 根据 sub_class_token_ids 找到 main_class_token_ids
	predict_sub_class_ids = config.classMapper.batch_find_main_class_ids_by_inaccurate_sub_class_ids(predict_sub_class_ids)

	# [(2372, 3442), ...] -> ['酒店', ......]
	return list(map(lambda ids: ''.join(config.classMapper.ids_to_words(ids)), predict_sub_class_ids))

# text_list = [
# 	'天台很好看，躺在躺椅上很悠闲，因为活动所以我觉得性价比还不错，适合一家出行，特别是去迪士尼也蛮近的，下次有机会肯定还会再来的，值得推荐',
# 	'环境，设施，很棒，周边配套设施齐全，前台小姐姐超级漂亮！酒店很赞，早餐不错，服务态度很好，前台美眉很漂亮。性价比超高的一家酒店。强烈推荐',
# 	"物流超快，隔天就到了，还没用，屯着出游的时候用的，听方便的，占地小",
# 	"福行市来到无早集市，因为是喜欢的面包店，所以跑来集市看看。第一眼就看到了，之前在微店买了小刘，这次买了老刘，还有一直喜欢的巧克力磅蛋糕。好奇老板为啥不做柠檬磅蛋糕了，微店一直都是买不到的状态。因为不爱碱水硬欧之类的，所以期待老板多来点其他小点，饼干一直也是大爱，那天好像也没看到",
# 	"服务很用心，房型也很舒服，小朋友很喜欢，下次去嘉定还会再选择。床铺柔软舒适，晚上休息很安逸，隔音效果不错赞，下次还会来"
# ]
# result = predict(text_list, 'hard')
# result = predict(text_list, 'soft')
# print(result)
# ['酒店', '酒店', '洗浴', '水果', '酒店']