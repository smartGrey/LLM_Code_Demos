import pickle
import numpy as np


class ChatIntentClassifierModel(object):
    def __init__(self, config):
        super(ChatIntentClassifierModel, self).__init__()
        self.config = config

        load_model = lambda file_name:\
            pickle.load(open(f'{config.chat_intent_classifier_model_save_path}/{file_name}','rb'))
        self.vectorizer = load_model('vectorizer.pkl')
        self.lr_model = load_model('lr_model.pkl')
        self.gbdt_model = load_model('gbdt_model.pkl')

    def predict(self, text):
        word_vector = self.vectorizer.transform([text.lower()])

        proba_lr = self.lr_model.predict_proba(word_vector)
        proba_gdbt = self.gbdt_model.predict_proba(word_vector)

        label_id = int(np.argmax((proba_lr + proba_gdbt) / 2, axis=1))

        return self.config.get_chat_intent_classifier_label(label_id)



# from config import Config
# config = Config()
# model = ChatIntentClassifierModel(config)
# result = model.predict('如何是好?')
# diagnosis