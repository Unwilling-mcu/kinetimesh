FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" websockets     pydantic pydantic-settings asyncpg redis httpx     python-multipart python-dotenv

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
