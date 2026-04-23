FROM python:3.12.3-slim

WORKDIR /app

COPY docker/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY docker/src/ ./src/
COPY src/purchase_prediction.py ./src/purchase_prediction.py
COPY src/clf_modeling.py ./src/clf_modeling.py
COPY src/classification_data.py ./src/classification_data.py
COPY docker/model/ ./model/

WORKDIR /app/src

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "1", "--threads", "4", "server:wsgi"]
