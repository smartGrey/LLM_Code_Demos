from chat_intent_classifier.model import ChatIntentClassifierModel
from config import Config
from diagnosis_intent_classifier.model import DiagnosisIntentClassifierModel
from slot_filling_ner.model import SlotFillingNerModel
from py2neo import Graph


# 为了避免每次访问app都重新创建对象，所以在另外一个脚本创建，然后引入


config = Config()
chat_intent_classifier = ChatIntentClassifierModel(config)
diagnosis_intent_classifier = DiagnosisIntentClassifierModel(config, config.diagnosis_intent_classifier_model_save_path)
slot_filling_ner = SlotFillingNerModel(config, config.slot_filling_ner_model_save_path)
graph_db = Graph(
    profile="bolt://localhost:7687",
    auth=("neo4j", "general-judge-riviera-mozart-beyond-2404")
)