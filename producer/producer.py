import json
import time

import requests

while True:
    time.sleep(5)
    url = "http://service:5000/streaming_data"
    response = requests.get(url, stream=True)
    headers = {'Content-Type': 'application/json'}
    requests.post("http://service:5000/upload_data", data=json.dumps(response.json()), headers=headers, stream=True)
