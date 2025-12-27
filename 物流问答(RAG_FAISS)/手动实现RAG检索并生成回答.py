from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from rag_db_utils import db


# 这里演示如何手动实现，实际上用不到这个文件，而是用 ConversationalRetrievalChain 代替整个过程


def search_rag_db(question_text, k=2):
	# 搜索相关文档，只返回最相似的 2 个
	related_docs = db.similarity_search(question_text, k=k)

	# 打印匹配分数(只打印最匹配的前20个)
	# 这里是按照余弦相似度匹配的，分数越低表示越相似
	for i, (doc, score) in enumerate(db.similarity_search_with_score(question_text, k=5), start=1):
		print(f'score{'[匹配成功]' if i <= k else ''}: {score:.2f} doc: {doc.page_content}')
	print('-' * 20)

	# 将相关文档整理为一个\n连接的字符串
	return '\n'.join([doc.page_content.strip() for doc in related_docs]).replace("\n\n", "\n")
# "part1 \n part2 \n part3"


def create_prompt(question_text):
	# 从 rag 数据库查询相关信息
	related_docs_text = search_rag_db(question_text)
	print('related_docs_text:')
	print(related_docs_text)
	print('-' * 20)

	# 生成 prompt
	prompt = PromptTemplate(
		input_variables=['context', 'question'],
		template='基于以下已知信息，简洁和专业的来回答用户的问题。不允许在答案中添加编造成分。\n\n已知信息：\n{context}\n\n用户问题：\n{question}',
	)
	return prompt.format(context=related_docs_text, question=question_text)


# 搜索数据库，模型结合搜索结果，做出回答
def search_and_answer(question_text):
	model = Ollama(model="qwen2.5:7b")
	prompt = create_prompt(question_text)
	return model.invoke(prompt)

answer_text = search_and_answer('库存还有多少')
print('answer_text:', answer_text)

# score[匹配成功]: 0.39 doc: 当前库存量：1000件
# score[匹配成功]: 0.62 doc: 存储货物类型：电⼦产品
# score: 0.87 doc: 预计到达⽇期：2023-01-20
# score: 0.90 doc: 出发地：⼴州
# score: 0.95 doc: 当前位置：上海分拨中⼼
# --------------------
# related_docs_text:
# 当前库存量：1000件
# 存储货物类型：电⼦产品
# --------------------
# answer_text: 当前库存还有1000件电⼦产品。