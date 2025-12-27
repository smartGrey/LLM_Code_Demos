from pathlib import Path

class Config:
	root_dir = Path(__file__).parent  # 项目根目录路径
	device = 'mps'

	# mysql 配置
	mysql_config = {
		'host': '127.0.0.1',
		'port': 3306,
		'user': 'root',
		'password': '123456',
		'db': 'edu_rag',
	}

	# redis 配置
	redis_config = {
		'host': '127.0.0.1',
		'port': 6379,
		'db': 0,
		'password': '1234'
	}

	# milvus 配置
	milvus_config = {
		'host': '127.0.0.1',
		'port': 19530,
		'collection_name': 'edu_rag',
	}

	# LLM api 配置
	LLM_name = 'gpt-5-nano' # 问答时用的 LLM 模型
	assessment_LLM_name = 'gpt-3.5-turbo' # 评估用的 LLM 模型 (便宜一些)
	LLM_API_KEY = 'sk-sGAK4y6nzkRNOEN65DBoKjlX0B9UGnJuCZ9QOh5yIN3vTAAV'
	LLM_API_URL = 'https://api.chatanywhere.tech/v1'

	# 评估用的本地模型（ollama模型）
	assessment_local_model_name = 'qwen2.5:7b'
	assessment_local_model_url = 'http://localhost:11434'

	# pretrained_models
	BERT_model_path = root_dir / 'pretrained_models' / 'bert-base-chinese'
	BGE_M3_model_path = root_dir / 'pretrained_models' / 'bge-m3'
	BGE_RERANKER_LARGE_model_path = root_dir / 'pretrained_models' / 'bge-reranker-large'

	# primary_query_strategy_selector
	intent_classification_checkpoint_model_save_path = root_dir / 'pretrained_models' / 'primary_query_strategy_selector' / 'model' / 'checkpoint_models'
	intent_classification_final_model_save_path = root_dir / 'pretrained_models' / 'primary_query_strategy_selector' / 'model' / 'final_model'
	intent_classification_final_model_report_save_path = root_dir / 'pretrained_models' / 'primary_query_strategy_selector' / 'model' / 'final_model' / 'report.txt'
	intent_classification_data_path = root_dir / 'pretrained_models' / 'primary_query_strategy_selector' / 'data.json'

	# 文档切割
	PARENT_CHUNK_SIZE = 1200
	CHILD_CHUNK_SIZE = 300
	CHUNK_OVERLAP = 50

	# 检索
	RETRIEVAL_NUM = 2
	FINAL_CANDIDATE_NUM = RETRIEVAL_NUM*2 # 可能有多个子查询

	# embedding match
	unstructured_doc_dir = root_dir / 'embedding_match' / 'rag_documents'
	subjects = ['ai', 'java', 'test', 'ops', 'bigdata'] # 非结构化数据的大类

	# bm25 match
	bm25_data_path = root_dir / 'bm25_match' / 'mysql_data.csv'
	bm25_match_threshold = 0.85


	service_phone_number = '0123456789' # 客服电话

	# single api 配置
	api_host = '127.0.0.1'
	api_port = 8004
	api_url = f'http://{api_host}:{api_port}'

	# web app 配置
	web_host = '127.0.0.1'
	web_port = 8050
	web_url = f'http://{web_host}:{web_port}'
	frontend_page_home = './frontend_page' # frontend_page 与 app_server.py 的相对路径
	short_circuit_reply = [ # 短路快捷回复，不经过 rag 系统
		{
			"pattern": r"^(你好|您好|hi|hello)",
			"response": "你好！我是黑马程序员，专注于为学生答疑解惑，很高兴为你服务！"
		},
		{
			"pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
			"response": "我是黑马程序员，你的智能学习助手，致力于提供 IT 教育相关的解答！"
		},
		{
			"pattern": r"^(在吗|在不在|有人吗)",
			"response": "我在！我是黑马程序员，随时为你解答问题！"
		},
		{
			"pattern": r"^(干嘛呢|你在干嘛|做什么)",
			"response": "我正在待命，随时为你解答 IT 学习相关的问题！有什么我可以帮你的？"
		}
	]

# print(Config.root_dir)