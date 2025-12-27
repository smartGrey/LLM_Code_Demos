import torch
from bert_model_and_config import Config, Model
from train_eval_test_predict import final_test
import argparse


# 命令行参数解析
parser = argparse.ArgumentParser(description="中文分类任务")
parser.add_argument("--task", type=str, default='test-bert', help="choose a task: test-bert(default)")
args = parser.parse_args()

# cd /Users/liuzhuocheng/Desktop/AI学习笔记/博学谷课程/codes/bert
# python run.py --help
# python run.py --task test-bert

if __name__ == '__main__':
	if args.model == 'test-bert':
		# 在这里放训练、调用模型的代码
		config = Config()
		model = Model(config).to(Config().device)
		model.load_state_dict(torch.load(Config().save_model_path))
		final_test(model, config)
	else:
		raise ValueError("请输入正确的模型名称")