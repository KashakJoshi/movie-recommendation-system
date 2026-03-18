FROM python:3.11-slim

WORKDIR /app

# system deps (important for scikit-surprise)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project code
COPY api ./api
COPY pipeline ./pipeline
COPY mlartifacts ./mlartifacts



# start server
CMD gunicorn -k uvicorn.workers.UvicornWorker api.main:app --bind 0.0.0.0:$PORT