from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import uuid
from rag_system.config import Config
import uvicorn # 用于运行 FastAPI
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# /Users/liuzhuocheng/Desktop/AI学习笔记/项目代码/教育问答(RAG)/rag_system
from rag_system.rag_system import RAGSystem


# 本文件用来启动一个 server，给其它服务提供一个独立的 http 接口
# 直接运行下面命令即可启动：
#       python api_server.py
# 可在浏览器访问：http://localhost:8004/docs
# 或者直接调用接口：http://localhost:8004/query


app = FastAPI(title="RAG 系统 API", description="外部服务可以通过这个 single_api 进行调用")
rag_system = RAGSystem(Config)


@app.post("/query", description="流式查询接口，返回RAG系统生成的答案")
async def streaming_query(request: Request) -> StreamingResponse:
    # {
    #     "query": "什么是人工智能？",
    #     "subject": "ai", # 可选，用于学科过滤
    #     "session_id": "a1b2c3d4-...", # 可选，用于维护对话历史. 第一次由本 single_api 成
    # }
    # 测试：{"query": "什么是人工智能？", "subject": "ai"}

    # 获取参数
    body = await request.json()
    query = body.get("query", "").strip()
    subject = body.get("subject", None)
    session_id = body.get("session_id", None)

    # 第一次访问，生成 session_id
    if not session_id:
        session_id = str(uuid.uuid4())

    # 生成器函数
    def _yield_response():
        # 调用问答系统的核心 query 方法，返回生成器（每次产出一个 token）
        for chunk, is_complete in rag_system.streaming_query(
            query=query,
            subject=subject,
            session_id=session_id,
        ):
            message = {
                "chunk": chunk, # str
                "is_complete": is_complete, # bool
                "session_id": session_id, # str
            }
            yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"
            # 使用 SSE 格式：data: {json}\n\n
            # ensure_ascii=False 确保中文不被转义为 \uXXXX

    # 返回流式响应，媒体类型为 text/event-stream（SSE 标准）
    return StreamingResponse(
        _yield_response(), # 接收生成器函数
        media_type="text/event-stream",
    )

if __name__ == '__main__':
    uvicorn.run(app, host=Config.api_host, port=Config.api_port)
