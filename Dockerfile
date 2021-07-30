FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime as setup

RUN apt-get update

RUN pip install ipython

COPY lf ./aleph
COPY ./requirements.txt ./requirements.txt
RUN cat requirements.txt | grep -v aleph | grep -v torch >> requirements_slim.txt
RUN pip install --no-cache-dir -r requirements_slim.txt
