from milvus_model.hybrid import BGEM3EmbeddingFunction # 文档向量化
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder # 重排序
import hashlib


# 功能：实现文档的向量化(语义嵌入)、混合搜索、重排序


# 向量的存储、检索
class EmbeddingMatch:
	def __init__(self, config):
		self.config = config

		# pretrained_models
		self.reranker = CrossEncoder(config.BGE_RERANKER_LARGE_model_path)
		self.embedding_function = BGEM3EmbeddingFunction(
			config.BGE_M3_model_path,
			use_fp16=False, # 使用 32 位浮点数
			device=config.device,
		)

		self.dense_dim = self.embedding_function.dim["dense"] # 稠密向量维度(1024)
		# dim: {"dense": 1024, "sparse": 1000, 'colbert_vecs': 1024}

		# Milvus 客户端
		milvus_c = self.config.milvus_config
		self.client = MilvusClient(
			uri=f"http://{milvus_c['host']}:{milvus_c['port']}",
			collection_name=milvus_c['collection_name'],
		)

		self._load_collection() # 创建 collection, 并载入内存


	# 创建 collection, 并载入内存
	def _load_collection(self):
		collection_name = self.config.milvus_config['collection_name']

		# 如果已有collection，则直接加载到内存
		if self.client.has_collection(collection_name):
			self.client.load_collection(collection_name)
			return

		# 创建 Schema
		schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
		schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100) # 存储 HASH str 作为 id
		schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535) # 子块内容
		schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
		schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
		schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535) # 父块内容
		schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=50) # 学科类别
		schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=300)

		# 设置 Index
		index_params = self.client.prepare_index_params()
		index_params.add_index(
			field_name="dense_vector",
			index_name="dense_index",
			index_type="IVF_FLAT", # 聚类
			metric_type="IP", # 内积
			params={"nlist": 128}, # 聚类中心数
		)
		index_params.add_index(
			field_name="sparse_vector",
			index_name="sparse_index",
			index_type="SPARSE_INVERTED_INDEX", # milvus 独有的“稀疏倒排索引”
			metric_type="IP",
			params={"drop_ratio_build": 0.2}, # 查询向量中最小的20%的向量值，将在搜索时被忽略
		)

		# 创建 collection
		self.client.create_collection(
			collection_name=collection_name,
			schema=schema,
			index_params=index_params,
		)

		self.client.load_collection(collection_name) # 加载到内存


	@staticmethod
	def _sparse_to_dict(sparse_vector):
		# 将 sparse 转为 dict, 以适应 Milvus 的 sparse_vector 输入
		return {i: value for i, value in zip(sparse_vector.col, sparse_vector.data)}


	def close(self):
		self.client.close()


	# 将文档添加到向量数据库
	def add_documents(self, documents): # documents: 切分后的子块
		if not documents: return # embedding_function 不接受空列表

		# 将文档文本转为嵌入向量
		embedding_vectors = self.embedding_function([doc.page_content for doc in documents])
		# {"dense": [[...], ...], "sparse": [[...], ...]}

		data = []
		for i, document in enumerate(documents):
			data.append({
				"id": hashlib.md5(document.page_content.encode('utf-8')).hexdigest(),
				"text": document.page_content,
				"dense_vector": embedding_vectors["dense"][i],
				"sparse_vector": self._sparse_to_dict(embedding_vectors["sparse"][i]), # 当前文档的一维稀疏向量

				# 参数 documents 中必须得有下面三个字段:
				"parent_content": document.metadata["parent_content"],
				"subject": document.metadata['subject'],
				"file_path": document.metadata['file_path'],
			})

		# 插入数据, 根据 hash_id 进行不重复插入
		self.client.upsert(collection_name=self.config.milvus_config['collection_name'], data=data)


	# 混合搜索+重排序
	def hybrid_search_and_rerank(self, query_text, subject=None):
		# 对查询进行嵌入
		query_embeddings = self.embedding_function([query_text])
		query_vector_dense = query_embeddings["dense"][0]
		query_vector_sparse = self._sparse_to_dict(query_embeddings["sparse"][0])

		# 创建稠密和稀疏向量搜索请求
		filter_expr = f"subject == '{subject}'" if subject else ""
		dense_request = AnnSearchRequest(
			data=[query_vector_dense],
			anns_field="dense_vector",
			param={"metric_type": "IP", "params": {"nprobe": 10}},
			limit=self.config.RETRIEVAL_NUM,
			expr=filter_expr
		)
		sparse_request = AnnSearchRequest(
			data=[query_vector_sparse],
			anns_field="sparse_vector",
			param={"metric_type": "IP", "params": {}},
			limit=self.config.RETRIEVAL_NUM,
			expr=filter_expr
		)

		# 混合搜索 + 第一次重排序
		hybrid_search_result_entities = self.client.hybrid_search(
			collection_name=self.config.milvus_config['collection_name'],
			reqs=[dense_request, sparse_request],
			ranker=WeightedRanker(1.0, 0.7),
			limit=self.config.RETRIEVAL_NUM,
			output_fields=["text", "parent_content", "subject", 'file_path',]
		)[0]
		# [{
		#     "id": "34fed0f1260c17d84bd21e7a89595f19",
		#     "distance": 0.5080522298812866,
		#     "entity": {
		#         "subject": "java",
		#         "text": "创建一个类BookTest......",
		#         "parent_content": "ava必须知道的300个问题......",
		#         "file_path": "..."
		#     }
		# },]

		# 获取去重后的父 documents
		parent_documents = self._get_unique_parent_documents(hybrid_search_result_entities)

		# 如果没有父 documents，或只有一个，则直接返回
		if not parent_documents or len(parent_documents) == 1:
			return parent_documents

		# 第二次重排序
		pairs_to_score = [[query_text, document.page_content] for document in parent_documents]
		scores = self.reranker.predict(pairs_to_score) # 计算得分 - [0.5, 0.3, 0.1, ...]

		# 根据得分从高到低进行排序, 返回父 documents
		ordered_result = sorted(zip(scores, parent_documents), reverse=True) # [(0.5, Document(...)), ...]
		return [e[1] for e in ordered_result][:self.config.FINAL_CANDIDATE_NUM]
		# [Document(
		# 	page_content='父块内容......',
		# 	metadata={
		# 		'subject': 'ai',
		# 		'file_path': '......',
		# 	},
		# ), ...]


	# 根据子块 hybrid_search_result_entities，获取去重后的父 documents
	@staticmethod
	def _get_unique_parent_documents(result_entities):
		parent_contents = set() # 去重后的父块内容
		result_parent_documents = [] # 合并后的文档列表

		for result_entity in result_entities: # 遍历所有子块
			entity = result_entity['entity']
			parent_content = entity['parent_content']

			if parent_content not in parent_contents: # 如果父块内容没有重复
				parent_contents.add(parent_content) # 加入 set
				result_parent_documents.append(
					Document(
						page_content=parent_content, # 父块内容
						metadata={ # 子块元数据
							'subject': entity['subject'],
							'file_path': entity['file_path'],
						},
					)
				)
		return result_parent_documents


# from config import Config
# embedding_match = EmbeddingMatch(Config)

# 添加数据
# from embedding_match.document_processor import process_documents
# documents = process_documents(Config)
# embedding_match.add_documents(documents)

# 搜索
# print(embedding_match.hybrid_search_and_rerank('什么是⻢尔科夫假设', 'ai'))

