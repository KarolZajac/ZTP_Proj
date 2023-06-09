FROM python:3.9.1

WORKDIR /app
COPY . /app

RUN pip install --upgrade -r requirements.txt


CMD ["python", "main.py"]
