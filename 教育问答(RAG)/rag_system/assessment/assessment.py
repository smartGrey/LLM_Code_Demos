from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision, # context_relevancy(原来的写法)两者是一回事
    context_recall
)
from datasets import Dataset
import json
from rag_system.config import Config


# 使用 ragas 对整个系统的 rag 效果进行评估


# 构建数据集
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)[:1]
eval_data = {
    # 提取每个数据条目的question字段，组成问题列表
    "question": [item["question"] for item in data],
    # 提取每个数据条目的answer字段，组成答案列表
    "answer": [item["answer"] for item in data],
    # 提取每个数据条目的context字段，组成上下文列表（每个context为列表）
    "contexts": [item["context"] for item in data],
    # 提取每个数据条目的ground_truth字段，组成真实答案列表
    "ground_truth": [item["ground_truth"] for item in data]
}
dataset = Dataset.from_dict(eval_data)


# 使用本地模型(需要先启动 ollama 并下载模型)
from langchain_ollama import OllamaEmbeddings, ChatOllama
llm = ChatOllama(model=Config.assessment_local_model_name, base_url=Config.assessment_local_model_url)
embeddings = OllamaEmbeddings(model=Config.assessment_local_model_name, base_url=Config.assessment_local_model_url)


# 使用在线 single_api
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# llm = ChatOpenAI(model=Config.assessment_LLM_name, api_key=Config.LLM_API_KEY, base_url=Config.LLM_API_URL)
# embeddings = OpenAIEmbeddings(api_key=Config.LLM_API_KEY)
# print(llm.invoke('你好')) # 测试网络是否通畅
# exit(0)


# 评估
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,  # 忠实度：答案是否基于上下文
        answer_relevancy,  # 答案相关性：答案与问题的匹配度
        context_precision,  # 上下文相关性：上下文是否仅包含相关信息
        context_recall  # 上下文召回率：上下文是否包含所有必要信息
    ],
    # 传入配置好的LLM模型
    llm=llm,
    embeddings=embeddings,
    batch_size=1,
)
print(result)
# {'faithfulness': 0.0000, 'answer_relevancy': 0.6759, 'context_precision': 0.0000, 'context_recall': 1.0000}






