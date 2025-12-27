import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


# 输入一个 logit list [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# 从分数最高的 k 个下标(token)中，按照给定的概率分数，有权重地随机抽一个返回
# 返回：tensor([7])
def logits_top_k_choose(logits, k=1):
	k = min(k, logits.shape[-1]) # k 不能大于词表大小
	k_top_logits, k_top_indexes = logits.topk(k, dim=-1)
	# k_top_logits：[0.9000, 0.8000]
	# k_top_indexes：[8, 7]

	# 随机抽取一个下标
	chosen_idx = torch.multinomial(F.softmax(k_top_logits, dim=-1), num_samples=1)
	# 需要先手动归一化，因为 multinomial 遇到负数会被当作极小正值，遇到全零/全负会报错。

	return k_top_indexes[chosen_idx]


class Bot:
	def __init__(self, config):
		self.config = config
		self.model = GPT2LMHeadModel.from_pretrained(config.model_path).to(config.device).eval()
		self.tokenizer = config.bert_tokenizer

		self.history = [] # 全部对话记录：token_ids_list - [[1,2,3], [4,5,6], [7,8,9]]

	def predict(self, user_input_text):
		user_input_tokens = self.tokenizer.encode(user_input_text, add_special_tokens=False)
		self.history.append(user_input_tokens)

		# 为模型准备生成所需要的上下文
		context_tokens = [self.tokenizer.cls_token_id]  # [102] - ['[CLS]']
		for token_ids in self.history[-self.config.context_memory_length:]:
			context_tokens.extend(token_ids)
			context_tokens.append(self.tokenizer.sep_token_id)  # 103 - [SEP]
		context_tokens = torch.tensor([context_tokens], dtype=torch.long, device=self.config.device)
		# tensor([[102, ..., 103, ..., 103]])

		# 模型开始生成，遇到[SEP]，则生成结束
		generated_tokens = []  # 根据context，生成的response
		for _ in range(self.config.generate_max_len):
			# 获取模型的输出分数
			logits = self.model(context_tokens).logits
			next_token_logits = logits[0, -1, :]  # [0.1, 0.2, 0.4, ...]

			# 对于已经生成过的token，对分数进行惩罚
			for token in set(generated_tokens):
				next_token_logits[token] /= self.config.repetition_penalty

			# 将输出'[UNK]'字符的概率设为无穷小，避免输出[UNK]
			next_token_logits[101] = -float('Inf')

			# 获取候选词 token
			next_token = logits_top_k_choose(next_token_logits, k=self.config.candidate_words_top_k)

			# 遇到 [SEP] 表明生成结束
			if next_token.item() == self.tokenizer.sep_token_id: break

			# 如果生成没有结束，将新生成的token添加到generated_tokens(生成结果)和context_tokens(上下文)中
			generated_tokens.append(next_token.item())
			context_tokens = torch.cat((context_tokens, next_token.unsqueeze(0)), dim=1)

		self.history.append(generated_tokens)  # 将本轮的回答结果加入 history
		generated_text = ''.join(self.tokenizer.convert_ids_to_tokens(generated_tokens))
		return generated_text


def cmd_interact(config):
	bot = Bot(config)

	# 存储聊天记录的文件
	with open(config.chat_log_path, 'a', encoding='utf8') as chat_log_file:
		while True:
			# 处理用户输入
			user_input = input("user: ") # 接收用户输入
			chat_log_file.write(f'{user_input}\n') # 记录到日志
			chat_log_file.flush()

			# 模型生成回答
			generated_text = bot.predict(user_input)

			# 在控制台输出并记录到日志
			print(f'bot: {generated_text}')
			chat_log_file.write(f'{generated_text}\n\n') # 记录到日志
			chat_log_file.flush()
# from config import Config
# cmd_interact(Config())