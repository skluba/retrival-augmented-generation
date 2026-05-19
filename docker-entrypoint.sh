#!/bin/sh
# Runtime starts as root so we can fix ownership of Docker volumes (e.g. rag_data → /app/data),
# then exec the process as the non-root `app` user (see Dockerfile).
set -e
if [ "$(id -u)" -eq 0 ]; then
    mkdir -p /app/data
    chown -R app:app /app/data
    exec /usr/sbin/runuser -u app -- "$@"
fi
exec "$@"
