import jieba
import fasttext
import torch
from flask import request, Flask, Response, json
from bert.train_eval_test_predict import predict
from bert.bert_model_and_config import Model, Config


# fasttext 准备工作
all_class = [x.strip() for x in open('../data/class.txt').readlines()]
# ['finance', 'realty', 'stocks', 'education', 'science', 'society', 'politics', 'sports', 'game', 'entertainment']
jieba.load_userdict('../data/stopwords.txt') # 把停用词表加载到用户词库，提高分词准确率
fasttext_model = fasttext.load_model('../models/fast_text_auto_tune.bin')


# bert 准备工作
bert_config = Config()
bert_model = Model(bert_config).to(Config().device)
bert_model.load_state_dict(torch.load(Config().save_model_path))


print('模型准备完成.')


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 解决json中文乱码



# fasttext 模型的两个请求

# 提交页面
@app.route('/fasttext-form', methods=["GET"], strict_slashes=False) # strict_slashes - 忽略末尾的斜杠
def get_fasttext_submit_page():
    with open('fasttext_submit_page.html', 'rb') as f:
        return f.read()

@app.route('/fasttext-classify', methods=["POST"], strict_slashes=False)
def fasttext_classify():
    text = request.form['content']

    # 分词
    input_text = ' '.join(jieba.lcut(text))
    # '公共英语 ( PETS ) 写作 中 常见 的 逻辑 词汇 汇总'

    res = fasttext_model.predict(input_text) # 预测
    # (('__label__3',), array([0.96119285]))

    class_idx = int(res[0][0][9:]) # 3
    class_name = all_class[class_idx] # education

    return Response(status=200, response=json.dumps({
        "Status": 'success',
        "Result": class_name,
    }))



# bert 模型的两个请求

# 提交页面
@app.route('/', methods=["GET"], strict_slashes=False) # 根路径也能访问
@app.route('/bert-form', methods=["GET"], strict_slashes=False)
def get_bert_submit_page():
    with open('bert_submit_page.html', 'rb') as f:
        return f.read()

@app.route('/bert-classify', methods=["POST"], strict_slashes=False)
def bert_classify():
    text = request.form['content']

    class_name = predict(bert_model, bert_config, text)

    return Response(status=200, response=json.dumps({
        "Status": 'success',
        "Result": class_name,
    }))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)