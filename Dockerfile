FROM python:3.11.6-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=src/app.py
EXPOSE 5000

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]