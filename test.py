import json

import requests

while True:
    url = "http://127.0.0.1:5000/streaming_data"
    response = requests.get(url, stream=True)
    headers = {'Content-Type': 'application/json'}
    requests.post("http://127.0.0.1:5000/upload_data", data=json.dumps(response.json()), headers=headers, stream=True)
