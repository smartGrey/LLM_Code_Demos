from pathlib import Path
from transformers import BertTokenizerFast


class Config:
    def __init__(self):
        self.device = 'mps'

        root_dir = Path(__file__).parent  # 项目根目录路径

        self.vocab_path = f'{root_dir}/model/vocab.txt'
        self.vocab_size = len(open(self.vocab_path, 'r', encoding='utf-8').read().split('\n')) # 13317
        self.bert_tokenizer = BertTokenizerFast(self.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
        self.padding_token = -100 # 计算 cross_entropy 时，忽略填充的 token(默认为-100)

        self.train_source_data_path = f'{root_dir}/data/medical_train.txt'
        self.valid_source_data_path = f'{root_dir}/data/medical_valid.txt'
        self.train_data_path = f'{root_dir}/data/medical_train.pkl'
        self.valid_data_path = f'{root_dir}/data/medical_valid.pkl'

        # 这三个是 HuggingFace 指定的文件结构
        self.model_path = f'{root_dir}/model/'
        self.model_config_path = f'{self.model_path}/config.json'
        self.model_bin_path = f'{self.model_path}/pytorch_model.bin'

        self.chat_log_path = f'{root_dir}/chat_log.txt' # 保存对话作为训练语料
        self.html_template_name = f'index.html'

        self.context_memory_length = 3 # 模型会记住这么多轮对话的内容
        self.generate_max_len = 300  # 每一轮问答中，模型回答的最大token长度
        self.repetition_penalty = 10.0 # 对于回复中的重复词，其预测分数除以这个值作为惩罚
        self.candidate_words_top_k = 4 # 对于回复的待选词，不是选择概率最高的，而是从前k个概率最高的词中随机抽取

        self.batch_size = 4
        self.epochs = 4
        self.lr = 2.6e-5
        self.eps = 1.0e-09

        self.max_grad_norm = 4.0 # 梯度裁剪的范数阈值
        self.gradient_accumulation_step_num = 4 # 累积多少次梯度之后，更新一次参数
        self.warm_up_proportion = 0.1 # 学习率预热阶段占整个训练阶段的比例
        # 使用Warmup预热学习率的方式，有助于使模型收敛速度变快
        # 先用最初的小学习率训练，然后每个step增大一点点，直到达到预设的学习率
        # 预热完成后的训练过程，学习率是衰减的