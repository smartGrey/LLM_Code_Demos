import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn import metrics


# predict_logits_list:      (batch, seq_len, vocab_size)
# mask_positions_list:      (batch, class_tokens_num)
#       mask 在句中的位置，用来从logits中取出对应的分数
#       [[15, 16], [23, 24], ...]
# sub_classes_ids_list: (batch, right_sub_classes_num, class_tokens_num)
#       所有正确答案下的子标签, right_sub_classes_num 长度不定
#       [[(2398, 3352), (2398, 3352)], (...), ...]
def loss_fn(predict_logits_list, mask_positions_list, sub_classes_ids_list, config):
    batch_total_loss = 0
    for (
	    logits, # (seq_len, vocab_size)
	    mask_positions, # (class_tokens_num)
	    sub_classes_ids, # (right_sub_classes_num, class_tokens_num)
    ) in zip(predict_logits_list, mask_positions_list, sub_classes_ids_list):
        mask_positions_logits_for_all_right_sub_classes = logits[mask_positions]\
            .repeat(len(sub_classes_ids), 1, 1)\
            .reshape(-1, config.vocab_size)
        # (class_tokens_num, vocab_size) - 找出 mask 位置的预测分数
        # (right_sub_classes_num, class_tokens_num, vocab_size) - 将 mask 位置的预测分数进行复制，以适配与多个子标签进行比较
        # (right_sub_classes_num * class_tokens_num, vocab_size) - 展平

        sub_classes_ids = torch.LongTensor(sub_classes_ids).reshape(-1)
        # 展平 - (right_sub_classes_num * class_tokens_num)

        loss = torch.nn.CrossEntropyLoss()(mask_positions_logits_for_all_right_sub_classes, sub_classes_ids)
        batch_total_loss += loss / len(sub_classes_ids) # 根据子标签数量进行平均

    return batch_total_loss / config.batch_size # 根据 batch_size 进行平均，得到损失数值


# 根据模型输出的 logits 和 mask 位置，得到预测的子标签
# predict_logits_list - (batch, seq_len, vocab_size)
# mask_positions_list - (batch, class_tokens_num)
def get_predict_sub_class_ids(predict_logits_list, mask_positions_list, config):
    # 展平 position
    all_mask_positions = mask_positions_list.flatten() # (batch * class_tokens_num) - [7,8, 7,8, 7,8]
    shift_positions = torch.tensor([config.input_max_seq_len])\
                    * torch.tensor(range(len(predict_logits_list))).repeat_interleave(2)
    # [0, 0, batch_i * input_max_seq_len, batch_i * input_max_seq_len, ...]
    # [0, 0, 512, 512, 1024, 1024, 1536, 1536, ...]
    all_mask_positions = all_mask_positions + shift_positions
    # [7, 8, 519, 520, 1031, 1032, 1543, 1544, ...]

    # 展平 logits
    all_logits = predict_logits_list.reshape(-1, config.vocab_size) # (batch * seq_len, vocab_size)

    # 抽取 MASK 位置的 logits
    all_mask_positions_logits = all_logits[all_mask_positions] # (batch * class_tokens_num, vocab_size)
    all_mask_positions_predict_tokens = all_mask_positions_logits.argmax(dim=-1) # (batch * class_tokens_num)
    return all_mask_positions_predict_tokens.reshape(-1, config.class_tokens_num) # (batch, class_tokens_num)


# 用来在 evaluate 时根据传入的 predict_words 和 target_words 计算指标
class Evaluator(object):
    def __init__(self, config):
        self.config = config
        # 展平保存
        self.target_main_classes_words_list = [] # ['电脑', '水果', ...]
        self.predict_main_classes_words_list = [] # ['电脑', '水果', ...]

    def add_batch_results_and_labels(self, target_main_classes_words_list, predict_main_classes_words_list):
        self.target_main_classes_words_list.extend(target_main_classes_words_list)
        self.predict_main_classes_words_list.extend(predict_main_classes_words_list)

    def compute_metrics(self, round_num=3):
        r = lambda num: round(num, round_num)
        results = (self.target_main_classes_words_list, self.predict_main_classes_words_list)

        main_classes_words = sorted(list(self.config.classMapper.main_class_to_sub_class_dict.keys()))
        # 用来给混淆矩阵确定顺序 - [(741, 5093), (2398, 3352), (2797, 3322), ...]
        confusion_matrix = metrics.confusion_matrix(*results, labels=main_classes_words)

        def calc_class_metrics(confusion_matrix, i):
            precision = confusion_matrix[i, i] / (sum(confusion_matrix[:, i]) + self.config.eps)
            recall = confusion_matrix[i, i] / (sum(confusion_matrix[i, :]) + self.config.eps)
            f1 = 2 * precision * recall / (precision + recall + self.config.eps)
            return {
                'precision': r(precision),
                'recall': r(recall),
                'f1': r(f1),
            }
        return {
            # 全局指标
            'accuracy': r(accuracy_score(*results)),
            'precision': r(precision_score(*results, average='weighted')),
            'recall': r(recall_score(*results, average='weighted')),
            'f1': r(f1_score(*results, average='weighted')),
            # 分类指标
            'class_metrics': {
                main_classes_words[i]: calc_class_metrics(confusion_matrix, i)
                for i in range(confusion_matrix.shape[0])
            }
        }