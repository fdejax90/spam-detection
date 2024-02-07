import logging

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()


# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def tokenize_words(text):
    logger.info("Tokenize words in text and remove punctuation")
    text = word_tokenize(str(text).lower())
    text = [token for token in text if token.isalnum()]
    return text


def remove_stopwords(text):
    logger.info("Remove stopwords from the text")
    text = [token for token in text if token not in stopwords.words("english")]
    return text


def stem(text):
    """Stem the text (originate => origin)"""
    logger.info("Stem the text with PorterStemmer (eg.: originate => origin)")
    text = [ps.stem(token) for token in text]
    return text


def transform(text):
    """Tokenize, remove stopwords, stem the text"""
    text = tokenize_words(text)
    text = remove_stopwords(text)
    text = stem(text)
    text = " ".join(text)
    return text


def nltk_pipeline(df):
    """Apply the transform function to the dataframe"""
    df["preprocessed_text"] = df["text"].apply(transform)
    return df


def encode_df(df, encoder=None):
    """Encode the features for training set"""
    is_fitted = hasattr(encoder, "vocabulary_") and encoder.vocabulary_ is not None
    if is_fitted:
        X = encoder.transform(df["preprocessed_text"]).toarray()
    else:
        X = encoder.fit_transform(df["preprocessed_text"]).toarray()
    y = df["label"].values
    return X, y, encoder
