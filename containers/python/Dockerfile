# Use an official Python runtime as a parent image
FROM python:3.11
USER root

# Install Pipenv
# --------------
ARG PIP_NO_CACHE_DIR=0
ARG PIPENVVERSION="2023.7.4"
RUN pip install pipenv==$PIPENVVERSION

# Install python libs
# -------------------
# Set the working directory in the container
WORKDIR /code

COPY src/ Pipfile Pipfile.lock ./
# Install project dependencies
RUN pipenv install --system --deploy
RUN [ "python3", "-c", "import nltk; nltk.download(['punkt', 'stopwords'], download_dir='/usr/local/nltk_data')" ]

