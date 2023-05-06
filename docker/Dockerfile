ARG PARENT_IMAGE
FROM $PARENT_IMAGE

WORKDIR /openrl

ADD setup.py setup.py
ADD openrl openrl
ADD README.md README.md

ENV VENV /root/venv

RUN \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN \
    python3 -m pip install --upgrade pip --no-cache-dir && \
    python3 -m pip install --no-cache-dir . && \
    python3 -m pip install --no-cache-dir ".[nlp]" && \
    rm -rf $HOME/.cache/pip

ENV PATH=$VENV/bin:$PATH

CMD /bin/bash
