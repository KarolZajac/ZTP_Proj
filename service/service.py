import csv
import sys
from datetime import datetime
import json
import random

from flask import Flask, render_template, request, Response, jsonify, redirect
import requests

from main import job

app = Flask(__name__)
training = False
uploading = False


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@singleton
class MetricHistory:

    def __init__(self):
        self.data_path = "bank.csv"
        self.train_metrics = {}
        self.test_metrics = {}
        self.training_times = []
        self.metrics_history = []
        self.from_date = "2000-06-09 14:30"
        self.to_date = "2040-06-09 14:30"

    def update(self, metrics):
        self.metrics_history.append(metrics)
        self.training_times.append(datetime.now())
        self.test_metrics = metrics[1]
        self.train_metrics = metrics[0]

    def set_time_section(self, from_date, to_date):
        self.from_date = fromCalendarToNormal(from_date)
        self.to_date = fromCalendarToNormal(to_date)


@app.route('/')
def home():
    from_date = MetricHistory().from_date
    to_date = MetricHistory().to_date

    indexes = []

    for idx, time in enumerate(MetricHistory().training_times):
        if to_date > datetimeToNormal(time) > from_date:
            indexes.append(idx)

    return render_template('index.html', train_metrics=MetricHistory().train_metrics,
                           test_metrics=MetricHistory().test_metrics,
                           history=zip([MetricHistory().training_times[i] for i in indexes],
                                       [MetricHistory().metrics_history[i] for i in indexes]))


@app.route('/filter', methods=['POST'])
def filter():
    MetricHistory().set_time_section(request.form.get("from"), request.form.get("to"))
    return redirect('/')


@app.route('/streaming_data', methods=['GET'])
def get_streaming_data():
    return generate_streaming_data()


@app.route('/upload_data', methods=['POST'])
def upload_data():
    global uploading
    global training
    if not training:
        uploading = True
        record = request.get_json()
        json_to_csv(record, 'bank.csv')
        uploading = False
        return "Success", 200
    else:
        return "Currently training model. Cannot upload data!", 500


@app.route('/retrain', methods=['POST'])
def retrain():
    global training
    global uploading
    if not uploading:
        training = True
        metrics = tuple(job())
        MetricHistory().update(metrics)
        print("Model retraining invoked!")
        training = False
        return redirect('/')
    else:
        return 'Cannot run training. Data is being uploaded!', 500


def fromCalendarToNormal(input_string):
    datetime_obj = datetime.strptime(input_string, "%Y-%m-%dT%H:%M")
    return datetime_obj.strftime("%Y-%m-%d %H:%M")


def datetimeToNormal(input_string):
    return input_string.strftime("%Y-%m-%d %H:%M")


def json_to_csv(json_data, csv_file):
    with open(csv_file, 'a+', newline='') as file:
        writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(json_data.values())


def get_column_contents(csv_file_path, column_name):
    column_contents = set()

    with open(csv_file_path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        header_names = [name.strip() for name in reader.fieldnames]

        if column_name not in header_names:
            print(f"Column '{column_name}' not found in the CSV file.")
            return column_contents

        for row in reader:
            if column_name in row:
                value = row[column_name]
                try:
                    value = int(value)
                except ValueError:
                    pass
                column_contents.add(value)

    return column_contents


def generate_streaming_data():
    csv_file_path = 'bank.csv'

    headers = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month",
               "duration", "campaign", "pdays", "previous", "poutcome", "y"]

    data = {}
    for i in headers:
        column_name = i
        unique_values = get_column_contents(csv_file_path, column_name)
        x = random.randint(0, len(unique_values) - 1)
        data[i] = list(unique_values)[x]

    return json.dumps(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
