FROM python:3.8

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install -U pip \
    && pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8000
