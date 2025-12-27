import uuid
from config import Config
from bm25_match.bm25_match import MySQLClient
from rag_match import RAGMatch


# 集成问答系统
# 主要实现：保存对话记录到数据库、为问答添加上下文、命令行问答
class RAGSystem:
    def __init__(self, config, verbose=False):
        self.config = config
        self.rag_matcher = RAGMatch(config, verbose=verbose)
        self.mysql_client = MySQLClient(config.mysql_config)
        self.init_conversation_table() # 初始化对话历史表

    # 初始化对话历史表
    def init_conversation_table(self):
        self.mysql_client.execute("""
            CREATE TABLE IF NOT EXISTS conversations(
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(36) NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                INDEX idx_session_id (session_id)
            )
        """)

    # 获取最近五轮对话
    def get_recent_history_by_session_id(self, session_id, limit=5):
        self.mysql_client.cursor.execute(
            """
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """,
            (session_id, limit),
        )
        # 将查询结果转换为字典列表
        history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
        return history[::-1] # 反转结果，按时间正序返回

    # 更新对话历史
    def update_history(self, session_id: str, question: str, answer: str, limit=5):
        self.mysql_client.execute( # 插入新的对话记录
            sql="""
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
            """,
            args=(session_id, question, answer)
        )
        self.mysql_client.execute( # 删除超出 5 轮的旧记录
            sql="""
                DELETE FROM conversations
                WHERE session_id = %s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id
                        FROM conversations
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ) AS sub
                )
            """,
            args=(session_id, session_id, limit)
        )

    # 清空对话历史
    def clear_history(self, session_id: str) -> bool:
        self.mysql_client.execute(
            sql="""
                DELETE FROM conversations
                WHERE session_id = %s
            """,
            args=(session_id,)
        )
        return True

    # 提问（流式，自动保存上下文）
    def streaming_query(self, query, session_id, subject=None):
        history = self.get_recent_history_by_session_id(session_id) if session_id else []

        # 流式返回回答
        answer = "" # 收集完整答案的字符串
        for chunk in self.rag_matcher.streaming_query(query, subject=subject, history=history):
            answer += chunk
            yield chunk, False

        self.update_history(session_id, query, answer) # 将完整问答存储到数据库

        yield "", True # 返回 True 表示流式生成结束

    def close(self):
        self.mysql_client.close()
        self.rag_matcher.close()

    def run_chat(self):
        session_id = str(uuid.uuid4())

        print(
            f'欢迎使用集成问答系统！会话ID: {session_id}.\n'
            f'支持的学科类别：{self.config.subjects}.\n'
            f'输入查询进行问答，输入 "exit" 退出.\n'
        )

        try:
            while True:
                query = input("输入问题: ").strip()

                if query.lower() == "exit":
                    print("再见！")
                    self.close()
                    return

                subject = input(f"请输入学科类别 ({'/'.join(self.config.subjects)}) (直接回车默认不过滤): ").strip()
                if subject not in self.config.subjects:
                    subject = None  # 如果学科过滤无效，忽略过滤

                # 迭代 query 方法的生成器
                for chunk, is_complete in self.streaming_query(query, session_id, subject=subject):
                    if is_complete:
                        print('\n------------回答结束--------------\n')
                    print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            self.close()
            print("已退出程序")


if __name__ == "__main__":
    RAGSystem(Config, verbose=True).run_chat()