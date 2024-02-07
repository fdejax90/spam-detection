"""Functions related to the training and evaluation pipeline"""

import logging
import time

import joblib
import pandas as pd
from preprocessing import train_val_test_split

# from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate

# plot_loss,
# plot_scores,
# save_scores,
from transforms import encode_df, nltk_pipeline
from utils import (
    SCORING,
    set_seed,
)
from xgboost.sklearn import XGBClassifier

logger = logging.getLogger()
logger.setLevel(logging.INFO)

model_version = "0.1.0"


MODELS = {
    # "NB": (MultinomialNB(), 1000),
    "XGBoost": (XGBClassifier(learning_rate=0.01, n_estimators=150), 2000),
}


def train_baselines(dataset_path, train_sizes, seed=0, test_set="test"):
    """Train all the baseline models."""
    set_seed(seed)

    for train_size in train_sizes:
        # Create list of metrics
        scores = pd.DataFrame(
            index=list(MODELS.keys()),
            columns=list(SCORING.keys()) + ["training_time", "inference_time"],
        )

        logger.info(f"Reading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        df = nltk_pipeline(df)
        df_train, df_val, df_test = train_val_test_split(
            df, train_size=train_size, has_val=True
        )

        # Cross-validate and test every model
        # Create a stratified k-fold object
        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for model_name, (model, unique_terms) in MODELS.items():
            # Encode the dataset
            encoder = TfidfVectorizer(max_features=unique_terms)
            X_train, y_train, encoder = encode_df(df_train, encoder)
            X_test, y_test, encoder = encode_df(df_test, encoder)
            #  breakpoint()

            # Evaluate model with cross-validation
            if test_set == "val":
                cv = cross_validate(
                    model,
                    X_train,
                    y_train,
                    scoring=list(SCORING.keys()),
                    cv=stratified_kfold,
                    n_jobs=-1,
                )
                for score_name, score_fn in SCORING.items():
                    scores.loc[model_name][score_name] = cv[f"test_{score_name}"].mean()

            # Evaluate model on test set
            if test_set == "test":
                start = time.time()
                model.fit(X_train, y_train)

                model_path = (
                    f"/opt/static/ml/{model_name}_version_{model_version}.joblib"
                )
                joblib.dump(model, model_path)
                logging.info(f"Saved model in {model_path}")
                encoder_name = encoder.__repr__()
                encoder_path = (
                    f"/opt/static/ml/{encoder_name}_version_{model_version}.joblib"
                )
                joblib.dump(encoder, encoder_path)
                logging.info(f"Saved encoder in {encoder_path}")
                end = time.time()
                scores.loc[model_name, "training_time"] = end - start

                start = time.time()
                y_pred = model.predict(X_test)
                end = time.time()
                scores.loc[model_name, "inference_time"] = end - start

                for score_name, score_fn in SCORING.items():
                    scores.loc[model_name, score_name] = score_fn(y_pred, y_test)

                # save_scores(
            #    ml_run, model_name, scores.loc[model_name].to_dict()
            # )

        # Display scores
        # plot_scores(ml_run, dataset_path.name)
        print(scores)
        print("-" * 100)
