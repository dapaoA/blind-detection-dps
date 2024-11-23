FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV TZ=Asia/Tokyo
ENV TERM=xterm-256color

RUN ln -fs /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

#### 0. Install python and pip
RUN apt-get -y update && apt-get install -y git wget curl && apt-get upgrade python3 -y && apt-get install python3-pip -y

#### 1. Install Pytorch
RUN pip3 install torch torchvision torchaudio ruff

#### 2. Install other dependencies
WORKDIR /usr/app
COPY . ./
RUN pip install -r ./requirements.txt
#### 4. change user
RUN useradd docker_user -u 1000 -m
