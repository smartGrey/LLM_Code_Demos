import os
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter
from tqdm import tqdm
from rag_system.config import Config
from rag_system.embedding_match.document_process.document_loaders.loaders import loaders
from rag_system.embedding_match.document_process.text_splitters.chinese_recursive_splitter import ChineseRecursiveTextSplitter


# 从预设的文档目录加载多种类型文件，同时添加元数据
def load_documents(config) -> list[Document]:
	result_documents = [] # 用于存储所有 document 对象
	supported_extensions = loaders.keys() # 所有支持的文件扩展名
	subject_dir_name_list = [name for name in os.listdir(config.unstructured_doc_dir) if name in config.subjects] # 获取所有学科根目录

	for subject_dir_name in tqdm(subject_dir_name_list, desc='正在加载并解析所有 RAG 文档...'): # 遍历所有学科根目录
		# 递归遍历每个学科目录下的所有目录
		for root_path, _dir_names, file_names in os.walk(config.unstructured_doc_dir / subject_dir_name):
			# 不递归地遍历每个目录下的所有文件
			for file_name in file_names:
				file_extension = os.path.splitext(file_name)[-1].lower() # 扩展名
				if file_extension not in supported_extensions: continue # 不支持的文件类型，跳过

				file_path = os.path.join(root_path, file_name) # 文件的完整路径
				loader_class = loaders[file_extension] # 根据扩展名获取对应的加载器类
				documents = loader_class(file_path).load() # 使用加载器加载文件
				for document in documents: # 为每个 document 对象添加元数据
					document.metadata["subject"] = subject_dir_name # 学科
					document.metadata["file_path"] = file_path # 文件的完整路径, 用于在向量数据库中定位原始文件
				result_documents.extend(documents)
	return result_documents
	# [Document(
	#   page_content='....',
	#   metadata={
	#       'subject': 'ai',
	#       'file_path': '....',
	#   }
	# )]


# 将文档切分为包含父块数据的子块
def process_documents(config) -> list[Document]:
	splitters = { # 初始化分割器[父块/子块]*[通用/markdown]
		"universal": {
			"parent": ChineseRecursiveTextSplitter(chunk_size=config.PARENT_CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP),
			"child": ChineseRecursiveTextSplitter(chunk_size=config.CHILD_CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
		},
		"markdown": {
			"parent": MarkdownTextSplitter(chunk_size=config.PARENT_CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP),
			"child": MarkdownTextSplitter(chunk_size=config.CHILD_CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP),
		}
	}
	child_chunk_documents = [] # 存储所有子块
	for document in tqdm(load_documents(config), desc='正在切分所有 RAG 文档...'): # 遍历每个原始文档
		file_extension = os.path.splitext(document.metadata.get('file_path', ''))[-1].lower() # 扩展名

		# 根据文件类型选择切分器
		file_type = 'markdown' if file_extension==".md" else 'universal'
		parent_splitter = splitters[file_type]["parent"]
		child_splitter = splitters[file_type]["child"]

		# 注意，切分时 metadata 会被继承

		for parent_chunk_document in parent_splitter.split_documents([document]): # 使用父块分词器切分文档并遍历
			for sub_chunk in child_splitter.split_documents([parent_chunk_document]): # 使用子块分词器切分父块并遍历
				sub_chunk.metadata["parent_content"] = parent_chunk_document.page_content # 在子块中记录父块的内容
				child_chunk_documents.append(sub_chunk)
	return child_chunk_documents
	# [Document(
	#   page_content='....',
	#   metadata={
	#       'subject': 'ai',
	#       'file_path': '....',
	#       'parent_content': '....',
	#   }
	# )]


if __name__ == '__main__':
	chunks = process_documents(Config)
	for chunk in chunks:
		print(chunk)
		print()
		print()