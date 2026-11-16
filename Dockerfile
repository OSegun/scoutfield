# The ScoutField demonstration as a container, for Google Cloud Run.
#
# The same app as every other tier: it reads the public artefact store, read-only and
# with no token, and runs both models with ONNX Runtime, so the image holds no PyTorch.
# Dependencies come from scoutfield/app/requirements.txt, the file the hosted app is
# tested with.
#
#   docker build -t scoutfield-demo .
#   docker run --rm -p 8080:8080 scoutfield-demo        # then open http://localhost:8080

# The Python version the dependency file was resolved and tested on.
FROM python:3.12-slim

# git: pip installs the pilot from its pinned git tag.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SCOUTFIELD_DEMO_SOURCE=store

WORKDIR /app

# Dependencies first, so a code change does not reinstall them.
COPY scoutfield/app/requirements.txt scoutfield/app/requirements.txt
RUN pip install -r scoutfield/app/requirements.txt

# Only what the app reads: the package and the configs (see .dockerignore).
COPY . .

# Run as an unprivileged user. The app caches store files under /app/.artifact_cache.
RUN useradd --create-home --uid 10001 demo && chown -R demo /app
USER demo

# Cloud Run passes the port to listen on in PORT; 8080 when run anywhere else.
EXPOSE 8080
CMD ["sh", "-c", "exec streamlit run scoutfield/app/streamlit_app.py --server.headless=true --server.address=0.0.0.0 --server.port=${PORT:-8080} --browser.gatherUsageStats=false"]
