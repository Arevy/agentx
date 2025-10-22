FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libopenblas-dev \
        wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/agent

COPY requirements.txt /workspace/agent/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /workspace/agent

CMD ["python", "agent_cli.py"]
