# This assumes the container is running on a system with a CUDA GPU
ARG version

FROM tensorflow/tensorflow:${version}

RUN apt install ca-certificates -y
RUN apt-get update && apt-get install -y \ 
    locales \
    curl htop \
    git \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install git -y && \
#     apt-get install ffmpeg libsm6 libxext6  -y && \
#     apt-get install htop -y

WORKDIR /home/ml_env_bridge

ARG USERNAME=softypo
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
#EXPOSE 8888

#ENTRYPOINT ["bash"]
RUN ["/bin/bash"]
#CMD ["/bin/sh", "-c", "while sleep 1000; do :; done"]