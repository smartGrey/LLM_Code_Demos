import pymysql
import pandas as pd
import redis
import json
import numpy as np
from rank_bm25 import BM25Okapi
import jieba


# 借助 MYSQL 和 REDIS 实现 BM25 模型：BM25Match


class MySQLClient:
    def __init__(self, mysql_config):
        self.connection = pymysql.connect(
            host=mysql_config['host'],
            user=mysql_config['user'],
            password=mysql_config['password'],
	        port=mysql_config['port'],
            database=mysql_config['db'],
        )
        self.cursor = self.connection.cursor() # 游标

    # 用于执行不需要返回的 sql 语句
    def execute(self, sql, args=None):
        self.cursor.execute(sql, args)
        self.connection.commit()

    def close(self): # 关闭数据库连接
        self.connection.close()
        # cursor 会自动关闭

# 导入数据到mysql
# from config import Config
# mysql_client = MySQLClient(Config.mysql_config)




# 封装连接、序列化
class RedisClient:
    def __init__(self, redis_config):
        self.client = redis.StrictRedis(
            host=redis_config['host'],
            port=redis_config['port'],
            password=redis_config['password'],
            db=redis_config['db'],
            decode_responses=True, # 默认返回字节，这里改为返回字符串
        )

    def set_data(self, key, value):
       self.client.set(key, json.dumps(value, ensure_ascii=False))

    def get_data(self, key):
       data = self.client.get(key)
       return json.loads(data) if data else None

    def close(self):
        self.client.close()

# from config import Config
# redis_client = RedisClient(Config.redis_config)




# 支持独立运行进行搜索
class BM25Match:
    def __init__(self, config):
        self.config = config
        self.redis_client = RedisClient(config.redis_config)
        self.mysql_client = MySQLClient(config.mysql_config)
        self.bm25_model = None
        self.original_questions = None # ['第一个问题', '第二个问题', ......]
        self.tokenized_questions = None # [['第一个', '问题'], ['第二个', '问题'], ......]
        self._init_bm25_model() # 需要先将问题固定地加载到 redis 中，防止数据库中数据被修改

    def _init_bm25_model(self):
        self.original_questions = self._fetch_all_questions() # 从 mysql 中加载数据
        self.tokenized_questions = [self._process_question(q) for q in self.original_questions] # 分词
        self.bm25_model = BM25Okapi(self.tokenized_questions) # 对所有 question 创建 BM25 模型

    @staticmethod
    def _process_question(text: str) -> list[str]: # 将问题文本转小写、分词，以便进行 BM25 匹配
        return jieba.lcut(text.lower())

    @staticmethod
    def _softmax(scores): # 对分数归一化
        scores = scores - np.max(scores) # 给所有分数都减去最大分数，防止指数过大，避免溢出
        exp_scores = np.exp(scores) # 求指数
        return exp_scores / exp_scores.sum() # 除以分母（总和）

    def _fetch_all_questions(self) -> list[str]: # 从数据库读取所有“问题”列
        self.mysql_client.cursor.execute("SELECT question FROM QA")
        return [r[0] for r in self.mysql_client.cursor.fetchall()] # 查出来是 tuple
        # ['问题1', '问题2', '问题3']

    def _fetch_answer(self, question) -> str: # 获取指定问题的答案
        self.mysql_client.cursor.execute("SELECT answer FROM QA WHERE question=%s", (question,))
        result = self.mysql_client.cursor.fetchone()
        return result[0] if result else None

    # 将数据从文件加载到 mysql
    def load_data_to_mysql(self):
        # 建表(如果不存在)
        self.mysql_client.execute('''
            CREATE TABLE IF NOT EXISTS QA ( -- QA
                id INT AUTO_INCREMENT PRIMARY KEY, -- 自增 id
                subject_name VARCHAR(20), -- 学科名
                question VARCHAR(1000), -- 问题
                answer VARCHAR(1000) -- 回答
            );
        ''')

        # 插入数据(这里可以优化，合并之后再 commit)
        for _, row in pd.read_csv(self.config.bm25_data_path).iterrows():
            self.mysql_client.execute(
                "INSERT INTO QA (subject_name, question, answer) VALUES (%s, %s, %s)",
                (row['学科名称'], row['问题'], row['答案'])
            )

    # 进行搜索，只返回最匹配的且达到阈值的一个
    def match(self, query: str) -> str:
        # 如果 redis 中有一模一样的问题，则直接返回答案
        redis_answer = self.redis_client.get_data(query)
        if redis_answer: return redis_answer

        # 对 bm25_model 中全部 question(来自mysql) 计算 bm25 分数
        tokenized_query = self._process_question(query) # 分词
        scores = self.bm25_model.get_scores(tokenized_query) # 计算分数
        scores = self._softmax(scores) # 对分数归一化
        best_match_index = np.argmax(scores) # 获取分数最大索引
        best_match_score = scores[best_match_index] # 获取分数最大值

        # 在 bm25_model 中匹配 question 失败
        if best_match_score < self.config.bm25_match_threshold: return ''

        # question 匹配成功：保存到 redis 中，返回答案
        question = self.original_questions[best_match_index] # 匹配到的“旧问题”
        answer = self._fetch_answer(question) # 根据匹配到的“旧问题”去 mysql 中获取“旧答案”
        self.redis_client.set_data(query, answer) # 将 “新query” 和 “旧答案” 作为问答对保存到 redis 中
        return answer

    def close(self):
        self.mysql_client.close()
        self.redis_client.close()

# from config import Config
# bm25_match = BM25Match(Config)
# bm25_match.load_data_to_mysql()
# print(bm25_match.match('如何创建单例对象')) # 输出：'......' (匹配的正确回答)
# print(bm25_match.match('如这是何干扰创建单信息例对象')) # 输出：'' (答案为空)