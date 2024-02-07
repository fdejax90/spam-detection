"""
Model / Encoder to be used in the API for predict endpoint
"""

MODEL_PATH = "/code/data/models"
MODEL_VERSION = "0.1.0"
# MODEL = {"LLM": "mshenoda/roberta-spam"}
MODEL = {"Scikit": f"{MODEL_PATH}/XGB_version_{MODEL_VERSION}.joblib"}
ENCODER_PATH = f"{MODEL_PATH}/TfidfVectorizer_version_{MODEL_VERSION}.joblib"
