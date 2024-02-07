import logging
from dataclasses import dataclass

import joblib
import pandas as pd

# import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .transforms import nltk_pipeline

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def llm_pipeline(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.config.id2label = {0: "non-spam", 1: "spam"}
    model.config.label2id = {"non-spam": 0, "spam": 1}
    return pipeline(task="text-classification", model=model, tokenizer=tokenizer)


@dataclass
class BaselinePipeline:
    sms: str

    def encode_df(self, encoder_path: str = "TfidfVectorizer().joblib"):
        """Encode the features for inference"""
        sms = pd.DataFrame([self.sms], columns=["text"])
        sms_df = nltk_pipeline(sms)
        encoder = joblib.load(encoder_path)
        self.encoded_sms = encoder.transform(sms_df["preprocessed_text"]).toarray()
        return self

    def predict(self, model_path: str = "XGBoost.joblib"):
        model = joblib.load(model_path)
        return model.predict(self.encoded_sms)
