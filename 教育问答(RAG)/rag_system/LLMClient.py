from openai import OpenAI


# 实现对线上 LLM 模型的流式/非流式请求


class LLMClient:
	def __init__(self, config):
		self.client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_API_URL)
		self.config = config

	def non_streaming_request(self, query) -> str:
		completion = self.client.chat.completions.create(
			model=self.config.LLM_name,
			messages=[
				{"role": "system", "content": "你是一个有用的助手。"},
				{"role": "user", "content": query},
			],
		)
		return completion.choices[0].message.content

	def streaming_request(self, prompt):
		completion = self.client.chat.completions.create(
			model=self.config.LLM_name,
			messages=[
				{"role": "system", "content": "你是一个能根据用户需求做出准确回复的助手"},
				{"role": "user", "content": prompt},
			],
			timeout=30,
			stream=True,
		)
		for chunk in completion:
			if chunk.choices and chunk.choices[0].delta.content:
				yield chunk.choices[0].delta.content

	def close(self):
		self.client.close()


if __name__ == '__main__':
	from config import Config
	llm_client = LLMClient(Config)

	# result = llm_client.non_streaming_request('如何创建单例对象')

	for chunk in llm_client.streaming_request('给我讲一个小故事'):
		print(chunk, end='')