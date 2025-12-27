# 说明
- (除了对 ocr.py 文件的引用路径)不要动这个目录下的文件，直接从 loaders.py 引用就行
- 这里其它文件都依赖ocr.py
- 其它每个包都有.load()方法，用于读取文件
- 可能需要安装的依赖包：
  - rapidocr_onnxruntime
  - python-pptx
  - pyMuPDF
  - python-docx