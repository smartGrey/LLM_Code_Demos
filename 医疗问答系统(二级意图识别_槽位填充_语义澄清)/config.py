from pathlib import Path
from transformers import BertTokenizer
from slot_filling_ner.data_utils import load_data_and_vocab


class Config:
	def __init__(self):
		self.device = 'mps'
		root_dir = Path(__file__).parent



		# 第一个 对话意图分类(chat_intent_classifier) 模型
		self.chat_intent_classifier_root = f'{root_dir}/chat_intent_classifier'
		self.chat_intent_classifier_model_save_path = f'{self.chat_intent_classifier_root}/models'
		self.chat_intent_classifier_data_path = f'{self.chat_intent_classifier_root}/labeled_data.txt'
		# id<->label
		self.chat_intent_classifier_id_to_label = ['greet', 'goodbye', 'reject', 'is_bot', 'confirm', 'diagnosis'] # id 从 0 开始
		# greet-问候、goodbye-告别、is_bot-用户问是不是机器人
		# diagnosis-用户希望进入诊断流程
		# confirm-用户同意澄清的内容
		# reject-用户不接受(同意)回复的内容
		self.chat_intent_classifier_label_to_id = {label: idx for idx, label in enumerate(self.chat_intent_classifier_id_to_label)}
		# get id/label
		self.get_chat_intent_classifier_label = lambda label_id: self.chat_intent_classifier_id_to_label[label_id]
		self.get_chat_intent_classifier_id = lambda label: self.chat_intent_classifier_label_to_id[label]



		# 第二个 医疗意图分类(diagnosis_intent_classifier) 模型
		self.diagnosis_intent_classifier_root = f'{root_dir}/diagnosis_intent_classifier'
		self.diagnosis_intent_classifier_data_root = f'{self.diagnosis_intent_classifier_root}/data'
		self.diagnosis_intent_classifier_train_data_path = f'{self.diagnosis_intent_classifier_data_root}/train.csv'
		self.diagnosis_intent_classifier_test_data_path = f'{self.diagnosis_intent_classifier_data_root}/test.csv'
		self.diagnosis_intent_classifier_model_save_path = f'{self.diagnosis_intent_classifier_root}/model/model.pth'
		# id<->label
		self.diagnosis_intent_classifier_id_to_label = [
			'定义', '病因', '预防', '临床表现(病症表现)', '相关病症', '治疗方法', '所属科室', '传染性', '治愈率',
			'禁忌', '化验/体检方案', '治疗时间', '其他'
		]
		self.diagnosis_intent_classifier_label_to_id = {
			label: idx for idx, label in enumerate(self.diagnosis_intent_classifier_id_to_label)
		}
		# get id/label
		self.get_diagnosis_intent_classifier_label = lambda label_id: self.diagnosis_intent_classifier_id_to_label[label_id]
		self.get_diagnosis_intent_classifier_id = lambda label: self.diagnosis_intent_classifier_label_to_id[label]
		# 超参
		self.diagnosis_intent_classifier_epochs = 10
		self.diagnosis_intent_classifier_lr = 2e-5
		self.diagnosis_intent_classifier_batch_size = 16
		self.diagnosis_intent_classifier_max_len = 60
		self.diagnosis_intent_classifier_class_num = len(self.diagnosis_intent_classifier_id_to_label) # 13
		# bert (这个需要先联网)
		self.diagnosis_intent_classifier_tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
		# 澄清阈值
		self.clarify_threshold = (0.4, 0.9) # 小于第一个数直接拒绝, 大于第二个数直接输出诊断结果，处于中间则进行澄清



		# 第三个 槽位填充NER(slot_filling_ner) 模型
		self.slot_filling_ner_root = f'{root_dir}/slot_filling_ner'
		self.slot_filling_ner_original_data = f'{self.slot_filling_ner_root}/data/train.json'
		self.slot_filling_ner_train_data = f'{self.slot_filling_ner_root}/data/train.txt'
		self.slot_filling_ner_model_save_path = f'{self.slot_filling_ner_root}/model_and_report/model.pth'
		self.slot_filling_ner_get_report_save_path = \
			lambda s: f'{self.slot_filling_ner_root}/model_and_report/report_{s}.txt'
		# id<->tag
		self.slot_filling_ner_tag_to_id = {"O": 0, "B_disease": 1, "I_disease": 2}
		self.slot_filling_ner_id_to_tag = {tag: idx for idx, tag in enumerate(self.slot_filling_ner_tag_to_id)}
		self.slot_filling_ner_tag_num = len(self.slot_filling_ner_tag_to_id) # 3
		# get tag/id
		self.slot_filling_ner_get_tag = lambda tag_id: self.slot_filling_ner_id_to_tag[tag_id]
		self.slot_filling_ner_get_tag_id = lambda tag: self.slot_filling_ner_id_to_tag[tag]
		# 超参
		self.slot_filling_ner_epochs = 10
		self.slot_filling_ner_lr = 2e-3
		self.slot_filling_ner_batch_size = 32
		self.slot_filling_ner_embedding_dim = 300
		self.slot_filling_ner_hidden_dim = 256
		self.slot_filling_ner_dropout = 0.2
		# 构建词表，得到下面的两个变量
		self.slot_filling_ner_word_to_id = {}
		self.slot_filling_ner_get_word_id = lambda word: self.slot_filling_ner_word_to_id.get(word, 1) # 1-UNK
		self.slot_filling_ner_vocab_size = None
		load_data_and_vocab(self)



		# 对话配置
		self.user_dialogue_context_save_path = f'{root_dir}/user_dialogue_context/username.json'
		self.preset_reply_sentences = {
			"greet": [
				"hi", "你好呀", "我是智能医疗诊断机器人，有什么可以帮助你吗",
				"hi，你好，你可以叫我小康", "你好，你可以问我一些关于疾病诊断的问题哦"
			],
			"goodbye": ["再见，很高兴为您服务", "bye", "再见，感谢使用我的服务", "再见啦，祝你健康"],
			"reject": ["很抱歉没帮到您", "I am sorry", "那您可以试着问我其他问题哟"],
			"is_bot": ["我是小康，你的智能健康顾问", "你可以叫我小康哦~", "我是医疗诊断机器人小康"],
		}
		self.semantic_schemes = { # 不同 diagnosis_intent 的语义策略
			"定义": {
				"slot_list" : ["Disease"], # 这个意图下只有一个槽位
				"slot_values": None, # 槽位的值默认是空的，需要 ner 后进行填充
				"cql_templates" : ["MATCH(p:Disease) WHERE p.name='{Disease}' RETURN p.desc"], # 查询知识图谱的查询模板
				"reply_template" : "'{Disease}' 是这样的：", # 常规流程下的回复模板, 后接 cql 查到的内容
				"clarify_template" : "您问的是 '{Disease}' 的定义吗？",
				"intent_strategy" : "", # 根据意图强度进行判断得到的回复策略(confirm/clarify/reject/'')
				"reject_text": "很抱歉没有理解你的意思呢~" # 用户不接受时的回复
			},
			"病因": {
				"slot_list": ["Disease"],
				"slot_values": None,
				"cql_templates": ["MATCH(p:Disease) WHERE p.name='{Disease}' RETURN p.cause"],
				"reply_template": "'{Disease}' 疾病的原因是：",
				"clarify_template": "您问的是疾病 '{Disease}' 的原因吗？",
				"intent_strategy": "",
				"reject_text": "您说的我有点不明白，您可以换个问法问我哦~"
			},
			"预防": {
				"slot_list": ["Disease"],
				"slot_values": None,
				"cql_templates": ["MATCH(p:Disease) WHERE p.name='{Disease}' RETURN p.prevent"],
				"reply_template": "关于 '{Disease}' 疾病您可以这样预防：",
				"clarify_template": "请问您问的是疾病 '{Disease}' 的预防措施吗？",
				"intent_strategy": "",
				"reject_text": "额~似乎有点不理解你说的是啥呢~"
			},
			"临床表现(病症表现)": {
				"slot_list": ["Disease"],
				"slot_values": None,
				"cql_templates": ["MATCH(p:Disease)-[r:has_symptom]->(q:Symptom) WHERE p.name='{Disease}' RETURN q.name"],
				"reply_template": "'{Disease}' 疾病的病症表现一般是这样的：",
				"clarify_template": "您问的是疾病 '{Disease}' 的症状表现吗？",
				"intent_strategy": "",
				"reject_text": "人类的语言太难了！！"
			},
			"相关病症": {
				"slot_list": ["Disease"],
				"slot_values": None,
				"cql_templates": ["MATCH(p:Disease)-[r:acompany_with]->(q:Disease) WHERE p.name='{Disease}' RETURN q.name"],
				"reply_template": "'{Disease}' 疾病的具有以下并发疾病：",
				"clarify_template": "您问的是疾病 '{Disease}' 的并发疾病吗？",
				"intent_strategy": "",
				"reject_text": "人类的语言太难了！！~"
			},
			"治疗方法": {
				"slot_list": ["Disease"],
				"slot_values": None,
				"cql_templates": ["MATCH(p:Disease) WHERE p.name='{Disease}' RETURN p.cure_way",
								"MATCH(p:Disease)-[r:recommand_drug]->(q:Drug) WHERE p.name='{Disease}' RETURN q.name",
								"MATCH(p:Disease)-[r:do_eat]->(q:Food) WHERE p.name='{Disease}' RETURN q.name"],
				"reply_template": "'{Disease}' 疾病的治疗方式、推荐药物、推荐食物有：",
				"clarify_template": "您问的是疾病 '{Disease}' 的治疗方法吗？",
				"intent_strategy": "",
				"reject_text": "没有理解您说的意思哦~"
			},
			"所属科室": {
				"slot_list": ["Disease"],
				"slot_values": None,
				"cql_templates": ["MATCH(p:Disease)-[r:cure_department]->(q:科室) WHERE p.name='{Disease}' RETURN q.name"],
				"reply_template": "得了 '{Disease}' 可以挂这个科室哦：",
				"clarify_template": "您想问的是疾病 '{Disease}' 要挂什么科室吗？",
				"intent_strategy": "",
				"reject_text": "您说的我有点不明白，您可以换个问法问我哦~"
			},
			"传染性": {
				"slot_list": ["Disease"],
				"slot_values": None,
				"cql_templates": ["MATCH(p:Disease) WHERE p.name='{Disease}' RETURN p.easy_get"],
				"reply_template": "'{Disease}' 较为容易感染这些人群：",
				"clarify_template": "您想问的是疾病 '{Disease}' 会感染哪些人吗？",
				"intent_strategy": "",
				"reject_text": "没有理解您说的意思哦~"
			},
			"治愈率": {
				"slot_list": ["Disease"],
				"slot_values": None,
				"cql_templates": ["MATCH(p:Disease) WHERE p.name='{Disease}' RETURN p.cured_prob"],
				"reply_template": "得了'{Disease}' 的治愈率为：",
				"clarify_template": "您想问 '{Disease}' 的治愈率吗？",
				"intent_strategy": "",
				"reject_text": "您说的我有点不明白，您可以换个问法问我哦~"
			},
			"治疗时间": {
				"slot_list": ["Disease"],
				"slot_values": None,
				"cql_templates": ["MATCH(p:Disease) WHERE p.name='{Disease}' RETURN p.cure_lasttime"],
				"reply_template": "疾病 '{Disease}' 的治疗周期为：",
				"clarify_template": "您想问 '{Disease}' 的治疗周期吗？",
				"intent_strategy": "",
				"reject_text": "很抱歉没有理解你的意思呢~"
			},
			"化验/体检方案": {
				"slot_list": ["Disease"],
				"slot_values": None,
				"cql_templates": ["MATCH(p:Disease)-[r:need_check]->(q:检查) WHERE p.name='{Disease}' RETURN q.name"],
				"reply_template": "得了 '{Disease}' 需要做以下检查：",
				"clarify_template": "您是想问 '{Disease}' 要做什么检查吗？",
				"intent_strategy": "",
				"reject_text": "您说的我有点不明白，您可以换个问法问我哦~"
			},
			"禁忌": {
				"slot_list": ["Disease"],
				"slot_values": None,
				"cql_templates": ["MATCH(p:Disease)-[r:not_eat]->(q:Food) WHERE p.name='{Disease}' RETURN q.name"],
				"reply_template": "得了 '{Disease}' 切记不要吃这些食物哦：",
				"clarify_template": "您是想问 '{Disease}' 不可以吃的食物是什么吗？",
				"intent_strategy": "",
				"reject_text": "额~似乎有点不理解你说的是啥呢~~"
			},
			"其他": {
				"slot_values": None,
				"final_reply_text": "非常抱歉，我还不知道如何回答您，我正在努力学习中~",
			}
		}