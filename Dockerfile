FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG INSTALL_GPT4ALL=false

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ "$INSTALL_GPT4ALL" = "true" ]; then pip install --no-cache-dir gpt4all; fi

COPY app ./app
COPY .env.example ./.env.example

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
