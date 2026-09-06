FROM python:3.12-slim-trixie

RUN apt update

RUN apt install -y --no-install-recommends git
RUN apt install -y --no-install-recommends nodejs
RUN apt install -y --no-install-recommends npm

RUN python3 -m pip install uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

ENV HARNESS_ROOT="/harness" \
	WORKSPACE_ROOT="/workspace" \
	REDIS_URL="redis://redis:6379/0" \
	COGNEE_MCP_URL="http://cognee-mcp:8000/mcp" \
	PLAYWRIGHT_MCP_URL="http://playwright:3080/mcp"

COPY README.md pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --no-editable

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

ENTRYPOINT ["uvicorn", "ripdude.server:app", "--host", "0.0.0.0", "--port", "8000"]