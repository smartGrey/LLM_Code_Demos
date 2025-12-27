import torch.nn as nn
from TorchCRF import CRF
from pathlib import Path
from data_utils import load_data_and_vocab


class Config(object):
    def __init__(self):
        self.device = 'mps'

        root_dir = Path(__file__).parent  # 项目根目录路径
        self.train_data_path = f'{root_dir}/train.txt' # 训练数据
        self.origin_data_root = f'{root_dir}/data_origin' # 原始数据路径
        self.model_path = f'{root_dir}/bilstm_crf_model.pt'
        self.report_path = f'{root_dir}/bilstm_crf_report.txt'
        self.name_to_label_dict =  {
            "治疗": "TREATMENT",
            "身体部位": "BODY",
            "症状和体征": "SIGNS",
            "检查和检验": "CHECK",
            "疾病和诊断": "DISEASE"
        }
        self.label_to_name_dict = {v: k for k, v in self.name_to_label_dict.items()}
        self.tag_to_id_dict = {
            "O": 0,
            "B-TREATMENT": 1,
            "I-TREATMENT": 2,
            "B-BODY": 3,
            "I-BODY": 4,
            "B-SIGNS": 5,
            "I-SIGNS": 6,
            "B-CHECK": 7,
            "I-CHECK": 8,
            "B-DISEASE": 9,
            "I-DISEASE": 10
        }
        self.id_to_tag_dict = {v: k for k, v in self.tag_to_id_dict.items()}
        self.tag_size = len(self.tag_to_id_dict) # 标签的种类个数

        self.embedding_dim = 300
        self.epochs = 5
        self.batch_size = 8
        self.hidden_dim = 256
        self.lr = 2e-3 # crf的时候，lr可以小点，比如1e-3
        self.dropout = 0.2

        # 构建词表，得到下面的两个变量
        self.word_to_id_dict = {}
        self.word_to_id = lambda word: self.word_to_id_dict.get(word, 1) # 1-UNK
        self.vocab_size = None
        load_data_and_vocab(self)


class NER_LSTM_CRF(nn.Module):
    def __init__(self, config):
        super(NER_LSTM_CRF, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim) # 词嵌入层
        self.lstm = nn.LSTM(
            self.config.embedding_dim,
            self.config.hidden_dim // 2, # 因为是双向，所以隐藏层维度要除以2，每个方向使用一半，最后输出层维度为hidden_dim
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.linear_output = nn.Linear(self.config.hidden_dim, self.config.tag_size) # 将输出维数映射到标签个数，输出结果是 发射分数
        self.crf = CRF(self.config.tag_size) # 这层训练结果为 转移分数

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