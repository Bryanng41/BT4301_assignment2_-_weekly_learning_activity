FROM python:3.12.3-slim

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "src/irisserver.py"]
