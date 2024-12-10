FROM python:3.10.6-buster
COPY skyai /skyai
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn skyai.api.fast:app --host 0.0.0.0 --port $PORT
