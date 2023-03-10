# This assumes the container is running on a system with a CUDA GPU
ARG version

# ARG USER_ID=1000
# ARG GROUP_ID=1000

FROM tensorflow/tensorflow:${version}

# RUN userdel -f www-data &&\
#     if getent group www-data ; then groupdel www-data; fi &&\
#     groupadd -g ${GROUP_ID} www-data &&\
#     useradd -l -u ${USER_ID} -g www-data www-data &&\
#     install -d -m 0755 -o www-data -g www-data /home/www-data

# USER www-data

# RUN apt install ca-certificates -y
# RUN apt-get update
# RUN apt-get install -y
# RUN locales \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /home/ml_env_bridge

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

#EXPOSE 8888

#ENTRYPOINT ["bash"]