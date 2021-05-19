FROM python:3.9-slim
COPY ./online_inference/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./online_inference .

CMD ["uvicorn", "prediction:app", "--host", "0.0.0.0", "--port", "8080"]