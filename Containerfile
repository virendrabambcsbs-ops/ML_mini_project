FROM docker.io/library/python:3.11-slim

LABEL org.opencontainers.image.title="QuickBite Market Basket Analysis"
LABEL org.opencontainers.image.description="Containerized Streamlit application deployed on RHEL using Podman"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        curl \
        iproute2 \
        less \
        procps \
        vim-tiny \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . /app

RUN useradd --create-home --shell /bin/bash mladmin \
    && mkdir -p /app/outputs \
    && chown -R mladmin:mladmin /app

USER mladmin

EXPOSE 8501

STOPSIGNAL SIGTERM

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]