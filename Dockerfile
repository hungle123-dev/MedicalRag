FROM python:3.12-slim AS backend
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY medrag_lab medrag_lab
COPY apps/__init__.py apps/__init__.py
COPY apps/api apps/api
COPY configs configs
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
