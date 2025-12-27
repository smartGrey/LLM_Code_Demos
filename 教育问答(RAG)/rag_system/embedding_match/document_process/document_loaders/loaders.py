from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from rag_system.embedding_match.document_process.document_loaders.doc_loader import OCRDOCLoader
from rag_system.embedding_match.document_process.document_loaders.img_loader import OCRIMGLoader
from rag_system.embedding_match.document_process.document_loaders.pdf_loader import OCRPDFLoader
from rag_system.embedding_match.document_process.document_loaders.ppt_loader import OCRPPTLoader

loaders = {
	".txt": lambda file_path: TextLoader(file_path, encoding="utf-8"),
	".pdf": OCRPDFLoader,
	".docx": OCRDOCLoader,
	".ppt": OCRPPTLoader,
	".pptx": OCRPPTLoader,
	".jpg": OCRIMGLoader,
	".png": OCRIMGLoader,
	".md": UnstructuredMarkdownLoader,
}