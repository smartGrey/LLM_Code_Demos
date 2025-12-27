import pickle
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# 固定随机种子
seed = 123
random.seed(seed)
np.random.seed(seed)


def load_data(config):
    # 读取文件
    with open(config.chat_intent_classifier_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    texts, label_ids = [], []
    for line in lines:
        text, label = line.strip().split(',')
        texts.append(text.lower())
        label_ids.append(config.get_chat_intent_classifier_id(label))

    return texts, label_ids
# (['如何是好', '不问了', ......], [5, 2, ......])
# 这里的数据是没有打乱的


def evaluate(lr_model, gbdt_model, x_test, y_test, config):
    y_predict_lr = lr_model.predict(x_test)
    y_predict_gbdt = gbdt_model.predict(x_test)

    y_predict_proba_lr = lr_model.predict_proba(x_test)
    y_predict_proba_gbdt = gbdt_model.predict_proba(x_test)
    y_predict_integrate = np.argmax((y_predict_proba_lr + y_predict_proba_gbdt) / 2, axis=1)

    print('---------- lr_model_report: -----------------')
    print(classification_report(y_test, y_predict_lr, target_names=config.chat_intent_classifier_id_to_label))
    print('---------- gdbt_model_report: ---------------')
    print(classification_report(y_test, y_predict_gbdt, target_names=config.chat_intent_classifier_id_to_label))
    print('---------- integrate_model_report: ----------')
    print(classification_report(y_test, y_predict_integrate, target_names=config.chat_intent_classifier_id_to_label))


def train(config):
    texts, labels = load_data(config)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, shuffle=True, random_state=seed
    )

    # 用 TF-IDF 进行词嵌入
    vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=0.0, max_df=0.9, analyzer='char')
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    # 创建并训练 LogisticRegression 模型
    lr_model = LogisticRegression(
        C=8, # 正则化强度的倒数, C 越小，正则化越强, 防止过拟合
        n_jobs=4, # 并行运行的 CPU 核心数，-1 表示使用所有核心
        max_iter=400, # 梯度下降时的最大迭代次数
        multi_class='ovr', # One-vs-Rest，为每个类别训练一个二分类
        random_state=seed # 随机种子
    )
    lr_model.fit(x_train, y_train)

    # 创建并训练 GBDT 模型
    gbdt_model = GradientBoostingClassifier(
        n_estimators=450, learning_rate=0.01, max_depth=8, random_state=seed
    )
    gbdt_model.fit(x_train, y_train)

    # 保存三个模型
    dump_model = lambda model, file_name:\
        pickle.dump(model, open(f'{config.chat_intent_classifier_model_save_path}/{file_name}','wb'))
    dump_model(vectorizer, 'vectorizer.pkl')
    dump_model(lr_model, 'lr_model.pkl')
    dump_model(gbdt_model, 'gbdt_model.pkl')

    # 评估
    evaluate(lr_model, gbdt_model, x_test, y_test, config)


# from config import Config
# train(Config())