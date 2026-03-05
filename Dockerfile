

FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim AS runtime

LABEL maintainer="SALIM Ayoub <ayoub.salim@pricepulse.ai>"
LABEL version="1.0.0"
LABEL description="PricePulse AI - Market Price Recommendation System"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*


COPY --from=builder /install /usr/local

RUN groupadd -r pricepulse && useradd -r -g pricepulse -d /app -s /sbin/nologin pricepulse


COPY --chown=pricepulse:pricepulse . .


RUN mkdir -p /app/reports /app/logs && \
    chown -R pricepulse:pricepulse /app/reports /app/logs

USER pricepulse

EXPOSE 8000


HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
