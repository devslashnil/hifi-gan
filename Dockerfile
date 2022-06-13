# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

# python3 inference.py --checkpoint_file ./LJ_V1/generator_v1
CMD [ "python3", "inference.py" , "--checkpoint_file", "./LJ_V1/generator_v1"]
