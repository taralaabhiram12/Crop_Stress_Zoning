FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DATA_DIR=/data SEED_DEMO=0
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd -m app && mkdir -p /data && chown -R app /app /data && chmod +x docker-entrypoint.sh
EXPOSE 8501
HEALTHCHECK --start-period=30s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://localhost:%s/_stcore/health'%os.environ.get('PORT','8501'))"
# Starts as root only to fix /data permissions, then runs Streamlit as user "app".
ENTRYPOINT ["./docker-entrypoint.sh"]
