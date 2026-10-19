# The organism's home. It starts when the container starts, keeps its memory in /data, and is restarted if it ever dies.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY deploy/requirements-live.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt
COPY lkm /app/lkm
ENV ORGANISM_HOME=/data PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
VOLUME /data
EXPOSE 8765
HEALTHCHECK --interval=60s --timeout=5s CMD curl -fs http://127.0.0.1:8765/health || exit 1
# binds all interfaces INSIDE the container; publish the port only to localhost or behind your own auth/TLS
CMD ["sh", "-c", "python -c \"import json,os;p=os.environ['ORGANISM_HOME']+'/habitat.json';c=json.load(open(p));c['host']='0.0.0.0';json.dump(c,open(p,'w'))\" && exec python -m lkm.live run --home $ORGANISM_HOME"]
