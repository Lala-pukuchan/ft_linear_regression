FROM python:3.9

WORKDIR /app

COPY ./requirements.txt .
RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y poppler-utils

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade wheel
RUN pip install sentence-transformers
RUN pip install huggingface_hub
RUN pip install autopep8
RUN pip install -r requirements.txt
