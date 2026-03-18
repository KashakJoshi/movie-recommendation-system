FROM python:3.11-slim

WORKDIR /app

# VERY IMPORTANT — install build tools for scikit-surprise
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install numpy==1.23.5
RUN pip install cython
RUN pip install scikit-surprise
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]