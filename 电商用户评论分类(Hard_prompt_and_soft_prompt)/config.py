from pathlib import Path
from transformers import AutoTokenizer
from utils.class_mapper import ClassMapper
from typing import Literal


class Config(object):
	def __init__(self, prompt_mode: Literal["hard", "soft"]):
		root_dir = Path(__file__).parent # 项目根目录路径
		self.device = 'mps'

		self.pretrained_model_name = 'google-bert/bert-base-chinese'
		self.train_data_path = f'{root_dir}/data/train.txt'
		self.dev_data_path = f'{root_dir}/data/dev.txt'

		self.best_model_save_path = f'{root_dir}/models/{prompt_mode}_prompt_best_model'
		self.best_model_report_save_path = f'{self.best_model_save_path}/report.txt'
		self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
		self.vocab_size = 21128

		self.input_max_seq_len = 512 # 输入的最大 token 序列长度, 根据样本长度分布来设置
		self.batch_size = 16
		self.lr = 5e-5
		self.warm_up_ratio = 0.06 # 线性预热阶段的比例
		self.epochs = 20
		self.logging_steps = 8
		self.eps = 1.0e-09


		# 这里的类别暂时都设计成[只能且必须]只是两个字
		self.class_tokens_num = 2 # 输出分类token的长度，也是所有类别的token的长度
		self.main_class_to_sub_class_dict = {
			'电脑': ['电脑'],
			'水果': ['水果'],
			'平板': ['平板'],
			'衣服': ['衣服'],
			'酒店': ['酒店'],
			'洗浴': ['洗浴'],
			'书籍': ['书籍'],
			'蒙牛': ['蒙牛'],
			'手机': ['手机'],
			'电器': ['电器'],
		}
		self.classMapper = ClassMapper(self)

		self.mask_token_text = '[MASK]'
		self.mask_token_texts = self.mask_token_text * self.class_tokens_num # '[MASK][MASK]'
		self.mask_token_id = self.tokenizer.convert_tokens_to_ids([self.mask_token_text])[0] # 103
		unused = lambda start, end: [f'[unused{i}]' for i in range(start, end+1)]
		unused_str = lambda start, end: ''.join(unused(start, end))
		# unused_str(1,5) -> '[unused1][unused2][unused3][unused4][unused5]'
		self.tokenizer.add_special_tokens({'additional_special_tokens': unused(1, 10)})
		# 加上这句'[unused1]'等 token 才能作为一个整体被识别，而不是被拆分成'[ unused1 ]'三部分

		self.generate_input_text_on_prompt_template = lambda original_text: {
			'hard': f'这是一条关于{self.mask_token_texts}的评论：{original_text}',
			# '这是一条关于[MASK][MASK]的评论：这个手机也太卡了。'

			'soft': f'{unused_str(1, 6)}{self.mask_token_texts}{unused_str(7, 10)}{original_text}',
			# '[unused1][unused2]......[unused6][MASK][MASK][unused7]......[unused10]这个手机也太卡了。'
		}[prompt_mode]
		# 这里两个模板的最终长度是相同的，方便对比效果
		# config.generate_input_text_on_prompt_template('这个手机也太卡了。')