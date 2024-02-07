# Builder image
FROM python:3.11-slim
USER root

# Install Pipenv
# --------------
ARG PIP_NO_CACHE_DIR=0
ARG PIPENVVERSION="2023.7.4"
RUN pip install pipenv==$PIPENVVERSION

# Install python libs
# -------------------
WORKDIR /code
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy
RUN [ "python3", "-c", "import nltk; nltk.download(['punkt', 'stopwords'], download_dir='/usr/local/nltk_data')" ]

#CMD pipenv install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


# The app
COPY ./app /code/app
EXPOSE 8080

ENV PYTHONUNBUFFERED=1
CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080" ]
