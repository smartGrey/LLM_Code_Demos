import torch
import torch.nn as nn
from transformers import BertModel


class DiagnosisIntentClassifierModel(nn.Module):
    def __init__(self, config, model_path=None):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained('google-bert/bert-base-chinese')
        self.linear = nn.Linear(768, config.diagnosis_intent_classifier_class_num)
        if model_path:
            self.load_state_dict(torch.load(model_path))

    def forward(self, inputs):
        output = self.bert(*inputs).pooler_output
        return self.linear(output)

    def predict(self, text):
        # 词嵌入编码
        inputs = self.config.diagnosis_intent_classifier_tokenizer.encode_plus(
            text,
            padding='max_length',
            truncation=True,
            max_length=60,
            return_tensors='pt'
        )

        # 用模型预测
        self.eval()
        with torch.no_grad():
            logits = self((inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]))

        # 处理预测结果
        probs = torch.softmax(logits, dim=-1) # 预测分数转为概率
        predict_label_idx = torch.argmax(probs, dim=-1).item() # 预测概率最大的标签id

        # 根据意图强度决定策略
        confidence = probs[0][predict_label_idx] # 意图强度
        low, high = self.config.clarify_threshold
        intent_strategy = 'confirm' if confidence > high else ('clarify' if confidence > low else 'reject')

        return {
            'name': self.config.get_diagnosis_intent_classifier_label(predict_label_idx),  # '临床表现(病症表现)'
            'confidence': confidence.item(), # 意图强度
            'intent_strategy': intent_strategy, # confirm/clarify/reject
            # 注意：这里的三种 strategy 和 chat_intent_classifier 输出的 6 种 intent 是两码事
        }
        # {'name': '临床表现(病症表现)', 'intent_strategy': 'confirm', confidence: 0.9}


# from config import Config
# config = Config()
# model = DiagnosisIntentClassifierModel(config, config.diagnosis_intent_classifier_model_save_path)
# print(model.predict('不同类型的肌无力症状表现有什么不同？'))
# {'name': '临床表现(病症表现)', 'intent_strategy': 'confirm'}