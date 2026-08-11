# syntax=docker/dockerfile:1.4
# openfilter-base = python:3.11-slim + all outstanding Debian security patches
# (rebuilt weekly): provides the PYTHONDONTWRITEBYTECODE/PYTHONUNBUFFERED env, the
# appuser account, and /app (WORKDIR) + /app/logs — so none of that is repeated here.
FROM plainsightai/openfilter-base:py3.11

# Copy model files
COPY ./filter_faceblur/model .

# Install pip + filter-faceblur at version from VERSION file
RUN --mount=type=bind,source=VERSION,target=/tmp/VERSION,ro \
    set -eux; \
    RAW="$(head -n1 /tmp/VERSION)"; \
    # strip optional leading v/V and whitespace
    PKG_VERSION="$(printf '%s' "$RAW" | tr -d ' \t\r\n' | sed 's/^[vV]//')"; \
    [ -n "$PKG_VERSION" ] || { echo "VERSION file is empty"; exit 1; }; \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      --index-url https://python.openfilter.io/simple \
      --extra-index-url https://pypi.org/simple \
      "filter-faceblur==${PKG_VERSION}"

# YuNetDetector auto-downloads the model into the installed package's
# model/weights/ dir at runtime, which fails under the non-root appuser
# because site-packages is owned by root. Pre-create the dir and hand
# ownership to appuser so any configured model URL can be fetched on first run.
RUN set -eux; \
    WEIGHTS_DIR="$(python -c 'import os, filter_faceblur; print(os.path.join(os.path.dirname(filter_faceblur.__file__), "model", "weights"))')"; \
    mkdir -p "$WEIGHTS_DIR"; \
    chown -R appuser:appuser "$WEIGHTS_DIR"

USER appuser
CMD ["python", "-m", "filter_faceblur.filter"]
