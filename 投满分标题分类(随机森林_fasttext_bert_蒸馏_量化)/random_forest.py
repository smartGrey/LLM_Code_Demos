from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# 读取数据
get_stop_words = lambda: open('./data/stopwords.txt').read().split() # 停用词
train_new_df = pd.read_csv('./data/train_new.csv', sep='\t')
labels = train_new_df['label']
texts = train_new_df['words'].values
# ic| texts: array(['中华 女子 学院 ： 本科 层次 仅 1 专业 招 男生',
#                   '两天 价 网站 背后 重重 迷雾 ： 做个 网站 究竟 要 多少 钱',
#                   ...,
#                   '连续 上涨 7 个 月   糖价 7 月 可能 出现 调整 行情'], dtype=object)


# 用 TF-IDF 构建词向量
vectorizer = TfidfVectorizer(stop_words=get_stop_words()) # 初始化 TF-IDF 向量化器
text_vectors = vectorizer.fit_transform(texts) # 将句子转换为 TF-IDF 矩阵
text_vectors.shape # (180000, 112420) - 稀疏矩阵 - 180000 个文档 * 112420 个词
text_vectors.nnz # 1247649 个非零元素
# 稀疏矩阵：
#   Coords(坐标对)	Values
#   (0, 14281)	0.42283884715188563
#   (0, 42367)	0.27922506928035273
#   ......
#   (1, 13753)	0.33246222996381825
#   (1, 87938)	0.5003666369136959
#   ......
#   (2, 73492)	0.4146752997754691
#   (2, 23973)	0.42417159339025456
#   ......


# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(text_vectors, labels, test_size=0.2, random_state=0)


# 实例化模型、训练、评估
model = RandomForestClassifier(n_estimators=10, random_state=0, verbose=10) # verbose=10 显示训练过程的详细程度
model.fit(train_x, train_y)
accuracy = model.score(test_x, test_y)
# 0.8018333333333333