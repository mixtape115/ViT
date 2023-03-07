FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

RUN DEBIAN_FRONTEND=noninteractive
RUN apt-get update 
ENV TZ=Asia/Tokyo
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    python3 \
    python3-pip \
    graphviz \
    opencv-data \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools

RUN python3 -m pip install numpy
RUN python3 -m pip install opencv-python
# RUN pip3 install -U pip wheel setuptools japanize-matplotlib
RUN pip3 install jupyter click numpy matplotlib seaborn pandas tqdm
# RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install visdom jupyter torchviz torchinfo japanize_matplotlib
ENV USER_NAME=user
ENV USER_UID=1000
ARG wkdir=/home/${USER_NAME}

RUN echo "root:root" | chpasswd \
    && useradd -m -u ${USER_UID} --groups sudo,video ${USER_NAME} \
    && echo "${USER_NAME}:${USER_NAME}" | chpasswd \
    && echo "%${USER_NAME}    ALL=(ALL)    NOPASSWD:    ALL" >> /etc/sudoers.d/${USER_NAME} \
    && chmod 0440 /etc/sudoers.d/${USER_NAME}

USER ${USER_NAME}
WORKDIR ${wkdir}


RUN sudo chown -hR ${USER_NAME}:${USER_NAME} ${wkdir}
