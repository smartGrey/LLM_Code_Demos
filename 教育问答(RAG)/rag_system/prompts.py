from langchain_core.prompts import PromptTemplate


# 这里管理整个项目的所有 Prompt


class Prompts:
	# 用于生成给用户的回复
	final_generate_answer_prompt = PromptTemplate(
		template='''
			你是一个智能助手，负责帮助用户回答问题。请按照以下步骤处理：
	
	        1. **分析问题和上下文**：
	           - 基于提供的上下文（如果有）和你的知识回答问题。
	           - 如果答案来源于检索到的文档，请在回答中明确说明，例如：“根据提供的文档，……”。
	
	        2. **评估对话历史**：
	           - 检查对话历史是否与当前问题相关（例如，是否涉及相同的话题、实体或问题背景）。
	           - 如果对话历史与问题相关，请结合历史信息生成更准确的回答。
	           - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和问题回答。
	
	        3. **生成回答**：
	           - 提供清晰、准确的回答，避免无关信息。
	           - 如果上下文和历史消息均不足以回答问题，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
	        **对话历史**:
	         {history}
	        **上下文**:
	         {context}
	        **问题**:
	         {user_query}
	
	        **回答**:
		''',
		input_variables=["context", "history", "user_query", "phone"],
	)

	# 用于让 AI 判断检索策略
	retrieval_strategy_prompt = PromptTemplate(
		template="""  
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索**
                * 说明：如果用户 query 的粒度适合直接进行检索，而且意图明确，容易在 RAG 知识库中进行匹配，则对用户查询直接进行检索，不进行任何增强处理。
                    * 示例：
                        * 查询：AI 学科学费是多少？
                        * 策略：直接检索
                    * 查询：JAVA的课程大纲是什么？
                        * 策略：直接检索
            2.  **假设问题检索**
                * 说明：如果用户的 query 由于看起来难以直接在 RAG 系统中进行匹配，但其需要的一些相关知识可能存在于数据库中，则需要使用 LLM 生成一个假设的引导答案，然后基于假设答案进行检索。
                    * 示例：
                        * 查询：人工智能在教育领域的应用有哪些？
                        * 策略：假设问题检索
            3.  **子查询检索**
                * 说明：如果用户的问题粒度过大，过于宽泛，涉及到多个方面，则需要将复杂的用户查询拆分为多个简单、具体的子查询，分别进行检索然后再合并结果。
                    * 示例：
                        * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                        * 策略：子查询检索
            4.  **回溯问题检索**
                * 说明：如果用户的问题过于具体、过于个性化、无用的细节过多，则需要将 query 转化为更基础、广泛、概括、更易于检索的问题，然后进行检索。
                    * 示例：
                        * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
		""",
		input_variables=["query"],
	)

	# 用于让 AI 生成 RAG 用到的引导答案
	guide_answer_prompt = PromptTemplate(
		template="""  
           假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
           问题: {query}  
           假设答案:  
        """,
		input_variables=["query"],
	)

	# 用于让 AI 生成 RAG 用到的对问题拆分后的子问题
	subquery_prompt = PromptTemplate(
		template="""  
			将以下复杂查询分解为多个简单子查询，每行一个子查询(用'\n'分割)，最多生成两个子查询（只保留生成的子查询问题，其它多余的文本都不需要）：
			举一个例子: 
			用户原始query：Milvus 和 Zilliz Cloud 在功能上有什么不同？
			子查询：Milvus 有哪些功能？\nZilliz Cloud 有哪些功能？
			
			这是用于的 query: {query}  
			子查询:  
		""",
		input_variables=["query"],
	)

	# 用于让 AI 生成 RAG 用到的对问题化简、回溯后的父问题
	backtracking_prompt = PromptTemplate(
		template="""  
			将以下复杂查询简化为一个更简单的问题：  
			查询: {query}  
			简化问题:  
		""",
		input_variables=["query"],
	)


if __name__ == '__main__':
	rga_prompt = Prompts.final_generate_answer_prompt
	result = rga_prompt.format(context="黑马程序员", history="", user_query="这个机构叫什么名称", phone="12345")
	print(result)