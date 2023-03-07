# This assumes the container is running on a system with a CUDA GPU
ARG version

FROM tensorflow/tensorflow:${version}

WORKDIR /ml_env_bridge

RUN apt-get update && apt-get install -y \
    locales \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

#EXPOSE 8888

#ENTRYPOINT ["python"]