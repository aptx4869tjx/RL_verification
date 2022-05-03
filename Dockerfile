FROM python:3.7-buster

RUN apt-get update
RUN apt-get install -y gcc python3-dev

WORKDIR /home

COPY . .

RUN cd /home/RL_verification && pip3 install --default-timeout=100 -r requirements.txt