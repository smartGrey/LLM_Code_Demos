from typing import List
from LLMClient import LLMClient
from prompts import Prompts
from primary_query_strategy_selector.primary_query_strategy_selector import PrimaryQueryStrategySelector
from bm25_match.bm25_match import BM25Match
from embedding_match.embedding_match import EmbeddingMatch


# 实现 RAG 系统中的检索功能
# 主要功能：融合BM25、嵌入向量搜索、初步意图识别、Query 改写

class RAGMatch:
	def __init__(self, config, verbose=False):
		self.config = config
		self.embedding_matcher = EmbeddingMatch(config)
		self.bm25_matcher = BM25Match(config)
		self.llm_client = LLMClient(config)
		self.prompts = Prompts
		self.primary_query_strategy_selector = PrimaryQueryStrategySelector(config)
		self.verbose = verbose
		self.query_adjust_strategies = {
			"直接检索": None, # 不需要调整问题
			"回溯问题检索": self.prompts.backtracking_prompt,
			"子查询检索": self.prompts.subquery_prompt,
			"假设问题检索": self.prompts.guide_answer_prompt,
		}
		if self.verbose: print('[RAG System] 初始化完成.')


	# query -> 直接检索/回溯问题检索/子查询检索/假设问题检索
	def _query_adjust_strategy_selector(self, query) -> str:
		template = Prompts.retrieval_strategy_prompt
		wrapped_query = template.format(query=query) # 用 prompt 包装 query
		return self.llm_client.non_streaming_request(wrapped_query) # 调用 LLM


	# 对用户的 query 进行调整，返回新的 query
	def _adjust_query(self, query) -> List[str]:
		strategy = self._query_adjust_strategy_selector(query)
		if self.verbose: print(f"[RAG System] RAG 检索策略：{strategy}")
		if strategy=='直接检索': return [query]

		if strategy == '直接检索':
			adjusted_query = [query]
		else:
			# 根据 prompt 构建 adjust query
			query = self.query_adjust_strategies[strategy].format(query=query)
			# 调用 LLM 进行 adjust
			adjusted_query = [self.llm_client.non_streaming_request(query).strip()]

			# 对子查询进行分割
			if strategy == "子查询检索":
				adjusted_query = [q.strip() for q in adjusted_query[0].split("\n")]

		if self.verbose: print(f"[RAG System] 调整后的 query：{' | '.join(adjusted_query)}")

		return adjusted_query


	# 根据多个 query 进行上下文检索，构建上下文(str)
	# 同时在 mysql 中根据 bm25进行搜索，在 milvus 中根据嵌入向量进行搜索
	def _retrieve_and_merge_context(self, query_list: List[str], subject=None) -> str:
		bm25_match_contents = [ # bm25 检索结果
			content
			for query in query_list
			for content in [self.bm25_matcher.match(query)]
			if content
		]
		embedding_match_contents = list({ # embedding_match 混合检索+重排序 结果, 去重
			document.page_content
			for query in query_list
			for document in self.embedding_matcher.hybrid_search_and_rerank(query, subject)
		})
		contents = bm25_match_contents + embedding_match_contents # 合并结果

		# 只取前5个结果
		cut_len = min(len(contents), self.config.FINAL_CANDIDATE_NUM)
		contents_cutted = contents[:cut_len]

		if self.verbose: print(f"[RAG System] 检索到：{len(contents)} 个文档. 截取后剩下：{cut_len} 个文档.")

		return '\n'.join(contents_cutted) # 拼接所有文档


	def close(self):
		self.llm_client.close()
		self.embedding_matcher.close()
		self.bm25_matcher.close()


	def streaming_query(self, user_query, subject=None, history=()):
		# 处理历史对话
		# history: [{'question': '...', 'answer': '...'}, ...]
		history = history[-5:]  # 只取最近5轮对话
		history_context = '\n'.join([
			f'历史问题{i}：{h["question"]}\n历史问题{i}的回答：{h["answer"]}'
			for i, h in enumerate(history, start=1)
		])
		# "历史问题1：...\n历史问题1的回答：...\n历史问题2：...\n历史问题2的回答：..."
		if self.verbose and history_context: print(f"[RAG System] 历史对话上下文：\n{history_context}")

		# 初步判断查询类型: 通用知识/专业咨询
		primary_query_strategy = self.primary_query_strategy_selector.intent_classification(user_query)
		if self.verbose: print(f"[RAG System] 初步判断查询策略：{primary_query_strategy}")

		# 检索 RAG 数据库并构建 context
		if primary_query_strategy == "通用知识": # 如果查询属于“通用知识”类别，则直接使用 LLM 回答
			merged_context = ''
		else:
			adjusted_query = self._adjust_query(user_query)
			merged_context = self._retrieve_and_merge_context(adjusted_query, subject=subject)

		# 构建最终的 query
		final_query = self.prompts.final_generate_answer_prompt.format(
			context=merged_context,
			history=history_context,
			user_query=user_query,
			phone=self.config.service_phone_number,
		)
		# 使用大模型获得流式输出结果
		for chunk in self.llm_client.streaming_request(final_query):
			yield chunk


if __name__ == '__main__':
	from config import Config
	rag_match = RAGMatch(Config, verbose=True)
	result = rag_match.streaming_query(user_query="AI学科的课程大纲内容有什么", subject="ai")
	for value in result:
		print(value, end="", flush=True)
