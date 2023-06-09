import csv
from datetime import datetime
import json
import random

from flask import Flask, render_template, request, Response, jsonify, redirect
import requests

from main import job

app = Flask(__name__)


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
        self.data_path = "./bank.csv"
        self.train_metrics = {}
        self.test_metrics = {}
        self.training_times = []
        self.metrics_history = []

    def update(self, metrics):
        self.metrics_history.append(metrics)
        self.training_times.append(datetime.now())
        self.test_metrics = metrics[1]
        self.train_metrics = metrics[0]


@app.route('/')
def home():
    return render_template('index.html', train_metrics=MetricHistory().train_metrics,
                           test_metrics=MetricHistory().test_metrics,
                           metrics_history=zip(MetricHistory().training_times, MetricHistory().metrics_history))


@app.route('/streaming_data', methods=['GET'])
def get_streaming_data():
    return generate_streaming_data()


@app.route('/upload_data', methods=['POST'])
def upload_data():
    record = request.get_json()
    json_to_csv(record, 'bank.csv')
    return "Success", 200


@app.route('/retrain', methods=['POST'])
def retrain():
    metrics = tuple(job())
    MetricHistory().update(metrics)
    print("Model retraining invoked!")
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)
