from fastapi import FastAPI

from .conf import ENCODER_PATH, MODEL
from .pipeline import BaselinePipeline, llm_pipeline
from .schema import PredictionOut, Sms

# some documentation in markdown
description = """  
## Documentation
**ℹ️ Read carefully before using**

This Software allows you to effectively identify and classify SMS messages 
as either 'spam' or 'not spam'.
We leverage an AI model that can analyze the content of an
SMS message and make a prediction based on its understanding of what constitutes spam.

"""

# create FastAPI app and load model
app = FastAPI(
    title="SMS Classification Spam / Non Spam",
    description=description,
    version="0.1",
    contact={
        "name": "Zilo",
        "url": "https://www.zilo.co.uk/",
        "email": "florian.dejax@gmail.com",
    },
)


# endpoint for healthcheck
@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


# create an endpoint /predict that receives POST requests and returns predictions
@app.post("/predict/", response_model=PredictionOut)
def predict(payload: Sms):
    if MODEL.get("LLM"):
        classifier = llm_pipeline(MODEL.get("LLM"))
        return classifier(payload.sms)[0]
    else:
        baseline_pipe = BaselinePipeline(payload.sms).encode_df(ENCODER_PATH)
        predictions = baseline_pipe.predict(MODEL.get("Scikit"))
        if 1 in predictions:
            return {"label": "spam"}
        else:
            return {"label": "non spam"}
