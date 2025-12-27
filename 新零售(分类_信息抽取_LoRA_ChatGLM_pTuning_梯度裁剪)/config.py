from pathlib import Path
from transformers import AutoTokenizer, AutoConfig


class Config:
	def __init__(self):
		self.device = 'mps'
		# self.device = 'cpu'
		root_dir = Path(__file__).parent  # 项目根目录路径
		self.best_model_save_path = f'{root_dir}/models/best_model'
		self.train_data_path = f'{root_dir}/data/train.jsonl'
		self.test_data_path = f'{root_dir}/data/test.jsonl'

		self.lora_rank = 8
		self.lora_alpha = 32
		self.grad_clip_norm = 0.5 # 梯度裁剪的梯度范数阈值
		self.batch_size = 4
		self.epochs = 2
		self.lr = 3e-5
		self.warmup_ratio = 0.06 # 学习率预热比例
		self.max_input_len = 150 # 模型最大输入长度(根据 2-sigma 得出)
		self.max_output_len = 156 # 模型最大输出长度(根据 2-sigma 得出)
		self.evaluate_steps = 1

		self.pretrained_model_path = f'{root_dir}/models/chatglm_6b_original'
		self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
		self.pretrained_model_config = AutoConfig.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
		# p-tuning 参数
		self.pretrained_model_config.prefix_projection = False # 是否使用前缀投影, 默认为False(p-tuning), 如果为True, 则为 p-tuning-v2
		self.pretrained_model_config.pre_seq_len = 200 # p-tuning 前缀长度

# config = Config()
# print(config.tokenizer.decode([0, 1, 2]))