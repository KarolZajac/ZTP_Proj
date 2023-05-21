FROM python:3.9.1

WORKDIR /app
COPY . /app
COPY crontab /etc/cron.d/crontab
RUN apt-get update && apt-get -y install cron

RUN pip install -r requirements.txt

RUN crontab /etc/cron.d/crontab
CMD cron && tail -f /dev/null
