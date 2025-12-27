from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# 模拟文档数据
# 这里实际开发中，应该用 PyMuPDFLoader 和 RecursiveCharacterTextSplitter 对文档进行读取和分割
documents = [
	Document(page_content="存储货物类型：电⼦产品"),
	Document(page_content="当前库存量：1000件"),
	Document(page_content="当前位置：上海分拨中⼼"),
	Document(page_content="预计到达⽇期：2023-01-20"),
	Document(page_content="出发地：⼴州"),
]


# 将文档加载到 RAG 数据库中
# 实际开发中应该将 db 保存到本地，然后从本地加载
db = FAISS.from_documents(
	documents,
	OllamaEmbeddings(model="qwen2.5:7b"),
)