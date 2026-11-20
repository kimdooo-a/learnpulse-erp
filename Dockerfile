FROM python:3.12-slim
# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && apt-get clean
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY learnpulse ./learnpulse
EXPOSE 8000
CMD ["python", "-m", "learnpulse", "up", "--host", "0.0.0.0", "--port", "8000"]
