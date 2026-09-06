FROM cognee/cognee:latest AS base

USER root
WORKDIR /app

RUN pip install uv
RUN uv pip install cognee_community_hybrid_adapter_falkor

RUN mkdir "adapter"

RUN cat <<'EOF' > /app/adapter/sitecustomize.py
from cognee_community_hybrid_adapter_falkor import register

EOF

ENV PYTHONPATH="/app/adapter:$PYTHONPATH"