FROM python:3.12.4-slim

WORKDIR /app

COPY ./app /app
COPY ./data /data
COPY ./requirements.txt /requirements.txt
COPY ./ml_model /ml_model

RUN pip install --no-cache-dir -r /requirements.txt

EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app/app.py"]
