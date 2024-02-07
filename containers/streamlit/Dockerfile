# Builder image
FROM python:3.11-slim
USER root

EXPOSE $STREAMLIT_PORT

# Install Pipenv
# --------------
ARG PIP_NO_CACHE_DIR=0
ARG PIPENVVERSION="2023.7.4"
RUN pip install pipenv==$PIPENVVERSION

# Install python libs
# -------------------
WORKDIR /app
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy

# The app
COPY app.py ./app.py

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV PYTHONUNBUFFERED=1

CMD [ "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
