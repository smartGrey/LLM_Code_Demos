import os
import random
from pprint import pprint
import streamlit as st
import json
from app_env import *


# 说明：
# 该脚本为响应时执行，即每次在网页发生交互，都会从头执行一遍该脚本
# 目前数据库中只记录了以下三类问题的相关数据：临床表现(病症表现)、治疗方法、禁忌
# 测试问题：
#   闲聊：
#       你好啊
#       不是的
#       是的
#       你是机器人吗？
#       再见
#   诊断：
#       心脏病如何治疗？
#       心脏病不能吃什么食物？
#       心脏病有什么症状？
#       支气管炎相关(需要澄清，但没有相关数据)
#       支气管炎看(需要澄清)


# 启动前准备
# 1. 需要配置 streamlit magic，避免网页打印 None 的问题
#       code ~/.streamlit/config.toml
#       添加两行内容：
#         [runner]
#         magicEnabled = false
# 2. 启动前需要连接 vpn 加载 transformers 的模型
# 3. 需要把 neo4j.dump 文件导入数据库中


# 启动方式：在终端运行
# streamlit run /Users/liuzhuocheng/Desktop/AI学习笔记/博学谷课程/医疗问答系统(二级意图识别_槽位填充_语义澄清)/app.py


# 要先运行数据库
# neo4j start
# neo4j console


# 查询数据库，返回一个字符串
def cqls_searcher(cql_list):
    results = []
    for cql in cql_list:
        result_list = [] # ['心脏病', ....]
        result_dict_list = graph_db.run(cql).data() # [{'p.cure_way': '心脏病'}, ......]
        if not result_dict_list: continue # []
        for result_dict in result_dict_list: # {'p.cure_way': '心脏病'/['心脏病']}
            result_values = list(filter(lambda x: x, result_dict.values())) # ['心脏病', ....] / [['心脏病', ....]] / []
            result_values and result_list.extend(result_values if isinstance(result_values[0], str) else result_values[0])
        results.append("、".join(result_list)) # '心脏病、心脏病、心脏病'
    return '\n'.join(results) # '心脏病、心脏病、心脏病\n......'


# 根据语义槽方案生成最终回答文本(完善 semantic_scheme dict)
def generate_answer(semantic_scheme):
    slot_kv = {'Disease': semantic_scheme['slot_values'][0]}
    strategy = semantic_scheme['intent_strategy']

    # 如果意图为拒绝，则直接返回预设回复
    if strategy == 'reject':
        semantic_scheme['final_reply_text'] = semantic_scheme.get('reject_text')
        return semantic_scheme

    # 查询数据库并生成回答
    cqls = [template.format(**slot_kv) for template in semantic_scheme['cql_templates']]
    db_answer = cqls_searcher(cqls) # text
    answer_text = (semantic_scheme['reply_template'].format(**slot_kv) + '\n' + db_answer) \
            if db_answer else "十分抱歉，知识库中暂未找到相关的知识。"
    # 这个是用来直接回复或者澄清后回复的答案

    if strategy == 'confirm':
        semantic_scheme['final_reply_text'] = answer_text
    elif strategy == 'clarify': # 需要进一步澄清，先把答案存起来
        semantic_scheme['final_reply_text'] = semantic_scheme['clarify_template'].format(**slot_kv)
        semantic_scheme['prepared_answer'] = answer_text

    return semantic_scheme


# 处理医疗逻辑(完善 semantic_scheme dict)
def medical_robot(text):
    # 医疗意图识别
    diagnosis_intent = diagnosis_intent_classifier.predict(text)
    # {'name': '临床表现(病症表现)', 'intent_strategy': 'confirm', confidence: 0.9}

    # 根据医疗意图找到对应的语义槽方案，并进行补全
    diagnosis_intent_name = diagnosis_intent['name']
    semantic_scheme = config.semantic_schemes[diagnosis_intent_name]
    semantic_scheme['intent_strategy'] = diagnosis_intent['intent_strategy']
    semantic_scheme['diagnosis_intent_name'] = diagnosis_intent_name
    semantic_scheme['diagnosis_intent_confidence'] = diagnosis_intent['confidence']

    # 疾病名称识别
    disease_entity = slot_filling_ner.predict(text) # ['心脏病']

    # 无法理解或处理的医疗问题
    if (not disease_entity) or (diagnosis_intent_name == "其他"):
        return semantic_scheme

    # 语义槽填充
    semantic_scheme['slot_values'] = disease_entity[:1] # ['心脏病']

    # 根据语义槽方案生成最终回答文本(完善 semantic_scheme dict)
    return generate_answer(semantic_scheme)


def dump_user_dialogue_context(data):
    with open(config.user_dialogue_context_save_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(
            data, sort_keys=True, indent=4,
            separators=(', ', ': '), ensure_ascii=False
        ))
def load_user_dialogue_context():
    path = config.user_dialogue_context_save_path
    if not os.path.exists(path):
        return {"prepared_answer": "hi，机器人小智很高心为您服务"}
    else:
        with open(path, 'r', encoding='utf8') as f:
            return json.loads(f.read())

render = lambda text: st.markdown(text.replace('\n', '<br>'), unsafe_allow_html=True)


def main():
    st.title("欢迎访问智能医疗问答系统")

    # 初始化会话状态，如果没有则创建
    st.session_state.history = st.session_state.get('history', [])

    # 在页面渲染对话历史
    for chat in st.session_state.history:
        if chat['role'] == 'user':
            with st.chat_message("user"):
                render(chat['content'])
        else:
            with st.chat_message("assistant"):
                render(chat['content'])

    # user_input 接收用户的输入
    user_input = st.chat_input("开始和我聊天吧～ ")
    if user_input:
        # 将用户问题加入到历史对话中，并在页面渲染
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            render(user_input)

        # 判断用户聊天意图
        user_intent = chat_intent_classifier.predict(user_input)
        print(f'---------------\n用户问题：{user_input}\n意图判断：{user_intent}')

        if user_intent in ["greet", "goodbye", "reject", "is_bot"]: # 闲聊意图时
            response_text = random.choice(config.preset_reply_sentences[user_intent]) # 直接随机返回设计好的答案
        elif user_intent == "confirm": # 澄清意图
            semantic_scheme = load_user_dialogue_context() # 加载之前生成好的答案
            response_text = semantic_scheme['prepared_answer']
        else: # 诊断意图
            semantic_scheme = medical_robot(user_input) # 进行医疗诊断
            pprint(semantic_scheme)
            if 'prepared_answer' in semantic_scheme:
                # 如果本次诊断结果为需要澄清，则会有 prepared_answer 字段，需要先保存
                dump_user_dialogue_context(semantic_scheme)
            response_text = semantic_scheme['final_reply_text']

        # 将回复信息加入到历史对话中，并在页面渲染
        st.session_state.history.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            render(response_text)

main()