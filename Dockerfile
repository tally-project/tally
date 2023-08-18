FROM wzhao18/tally:base

WORKDIR /home/tally

COPY . .

RUN cp ./tests/cudnn_samples_v8/mnistCUDNN/data . -r

RUN mkdir /etc/iceoryx && \
    cp config/roudi_config.toml /etc/iceoryx/roudi_config.toml

RUN cd /home/tally && \
    make