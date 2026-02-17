FROM python:3.11-slim

WORKDIR /app

COPY rag/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY rag/ .

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "app.py"]
