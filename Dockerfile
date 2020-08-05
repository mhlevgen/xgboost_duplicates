FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libffi-dev libpq-dev gcc

COPY ./ cian_task/
WORKDIR cian_task/
RUN pip install -r requirements.txt