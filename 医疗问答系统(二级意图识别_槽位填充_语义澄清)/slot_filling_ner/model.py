import random
import numpy as np
import torch
from torch import nn
from TorchCRF import CRF

# 固定随机种子
seed = 123
random.seed(seed)
np.random.seed(seed)


class SlotFillingNerModel(nn.Module):
    def __init__(self, config, model_path=None):
        super(SlotFillingNerModel, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(
            self.config.slot_filling_ner_vocab_size,
            self.config.slot_filling_ner_embedding_dim
        ) # 词嵌入层
        self.lstm = nn.LSTM(
            self.config.slot_filling_ner_embedding_dim,
            self.config.slot_filling_ner_hidden_dim // 2, # 因为是双向，所以隐藏层维度要除以2，每个方向使用一半，最后输出层维度为hidden_dim
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(self.config.slot_filling_ner_dropout)
        self.linear_output = nn.Linear(self.config.slot_filling_ner_hidden_dim, self.config.slot_filling_ner_tag_num) # 将输出维数映射到标签个数，输出结果是 发射分数
        self.crf = CRF(self.config.slot_filling_ner_tag_num) # 这层训练结果为 转移分数

        if model_path:
            self.load_state_dict(torch.load(model_path))

    def lstm_part(self, x, mask):
        x = self.embedding(x)
        x, _ = self.lstm(x) # lstm 不传入 hidden_state，会自动随机初始化
        x = self.dropout(x)
        x = self.linear_output(x)

        return x * mask.unsqueeze(-1) # 乘以 mask，防止填充位置的输出被计算
        # 最初生成 mask 的时候，x 还没有进行词嵌入，每个 word 还只是一维的 token，所以现在需要广播才能作用于 x
        # mask: [batch_size, seq_len] -> [batch_size, seq_len, 1] --广播--> [batch_size, seq_len, tag_size]

    def forward(self, x, mask): # 这个函数只用来进行推理预测，不用它的结果计算损失
        x = self.lstm_part(x, mask)
        return self.crf.viterbi_decode(x, mask)
        # 用 viterbi(动态规划) 计算最优路径(tag 序列)
        # 这里返回的是最优路径的 tag ids

    def loss_fn__log_likelihood(self, x, true_y, mask): # 用来优化模型的(对数似然)损失函数
        x = self.lstm_part(x, mask)
        return -self.crf(x, true_y, mask) # 通过 前向-后向算法，用动态规划的思路计算损失

    def predict(self, text):
        # 处理模型输入
        x_tokens = torch.tensor([self.config.slot_filling_ner_get_word_id(c) for c in text]).unsqueeze(0)
        mask = torch.ones(x_tokens.size(1)).bool().unsqueeze(0)

        # 用模型进行预测
        self.eval()
        with torch.no_grad():
            # 标注得到 tags
            tag_ids = self(x_tokens, mask)[0] # [0, 0, 1, 2, 2, ......]

        # 对单个字的标注结果进行整理
        entities = []
        iterator = [(c, tag_id) for c, tag_id in zip(text, tag_ids)]
        # '...常...' + [..., 2, ...] -> [('常', 2), ...]
        for word, tag_id in iterator:
            if tag_id == 1: # 实体开头
                entities.append(word) # '常'
            elif tag_id == 2 and entities: # 实体后部
                prev_word = entities[-1] # '血常'
                entities[-1] = prev_word + word # '血常规'

        return entities

# from config import Config
# config = Config()
# model = SlotFillingNerModel(config, model_path=config.slot_filling_ner_model_save_path)
# model.predict('您好，我想请问您一下啊，稳心颗粒可以适于房性早搏的症状吗？') # ['房性早搏']
# model.predict('医生苯巴比妥东莨菪碱片，有心脏病的话不可以吃吧?谢谢') # ['心脏病']
# model.predict('什么是直肠息肉?') # ['是直肠息肉']
# model.predict('为什么会有\"十男九痔\"的说法') # ['十男九痔']
# 目前只支持预测单个疾病实体，多个会识别出错