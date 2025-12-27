import requests
import json
from rag_system.config import Config


# 用来测试 api server


def try_api(query, source_filter=None, session_id=None):
    req_data = {
        "query": query,                # 用户的问题
        "source_filter": source_filter, # 学科过滤条件（可选）
        "session_id": session_id       # 会话 ID，用于维护对话历史 (第一次应该从后端生成传来)
    }
    with requests.post(Config.api_url+'/query', json=req_data, stream=True) as response:
        if response.status_code != 200:
            print(f"请求失败: {response.status_code} - {response.text}")
            return

        # 逐行读取服务器返回的流式响应（每行是一个 SSE 消息）
        for line in response.iter_lines(decode_unicode=True):
            line = line.strip()
            if line.startswith("data:"):
                json_str = line[5:].strip() # 提取 "data:" 之后的 JSON 字符串。剩下的应该是 json
                data = json.loads(json_str)
                print(data)


try_api("AI学科的课程内容有什么", source_filter="ai")
# {'chunk': '下面', 'is_complete': False, 'session_id': '168b5ee6-d891-43a6-8785-59e7e50a6225'}
# {'chunk': '是', 'is_complete': False, 'session_id': '168b5ee6-d891-43a6-8785-59e7e50a6225'}
# {'chunk': 'AI', 'is_complete': False, 'session_id': '168b5ee6-d891-43a6-8785-59e7e50a6225'}
# ............