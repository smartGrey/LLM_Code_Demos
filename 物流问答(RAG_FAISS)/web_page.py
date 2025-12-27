import streamlit as st
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from rag_db_utils import db


# 这里使用 ConversationalRetrievalChain + streamlit 实现

# 启动方式：
# cd /Users/liuzhuocheng/Desktop/AI学习笔记/博学谷课程/物流问答(RAG_FAISS)
# streamlit run web_page.py


# streamlit 渲染页面会反复执行这个脚本，这里仅第一次运行时加载
@st.cache_resource
def init_chain():
	llm = Ollama(model="qwen2.5:7b")
	return ConversationalRetrievalChain.from_llm(
		llm=llm,
		retriever=db.as_retriever(),
		memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
	)
qa_chain = init_chain()


# 设置页面标题
title = '物流助手'
st.set_page_config(page_title=title)
st.title(title)


# 初始化 UI 的会话状态
if "messages" not in st.session_state:
	st.session_state.messages = []


# 渲染历史聊天记录
for message in st.session_state.messages:
	role, content = message["role"], message["content"]
	with st.chat_message(role):
		st.markdown(content)


# 接收用户新的输入
if user_input := st.chat_input("请输⼊你的问题:"):
	# 当用户回车提交时，保存并显示用户的输入
	st.session_state.messages.append({"role": "user", "content": user_input})
	with st.chat_message("user"):
		st.markdown(user_input)

	# 调⽤模型获取回答
	result = qa_chain.invoke({"question": user_input})
	response_text = result["answer"]

	# 保存并显示模型回答
	st.session_state.messages.append({"role": "assistant", "content": response_text})
	with st.chat_message("assistant"):
		st.markdown(response_text)


# 显示检索到的文档（调试用）
if st.checkbox("显示相关文档"):
	if "messages" in st.session_state and len(st.session_state.messages) > 0:
		last_question = st.session_state.messages[-2]["content"]  # 最后一个用户问题
		docs = db.similarity_search(last_question, k=3)
		for i, doc in enumerate(docs, start=1):
			st.write(f"**文档 {i}**: {doc.page_content[:20]}")
		st.write('. . . . . .')