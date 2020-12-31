import base64
import datetime

import cv2
import numpy as np

from flask import Flask, Response
from kafka import KafkaConsumer



# Fire up the Kafka Consumer
topic = "raw-video"

consumer = KafkaConsumer(
    topic,
    bootstrap_servers=['kafka1:19091']
)


# Set the consumer in a Flask App
app = Flask(__name__)

@app.route('/video', methods=['GET'])
def video():
    """
    This is the heart of our video display. Notice we set the mimetype to
    multipart/x-mixed-replace. This tells Flask to replace any old images with
    new values streaming through the pipeline.
    """
    return Response(
        get_video_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame')


def get_video_stream():
    """
    Here is where we recieve streamed images from the Kafka Server and convert
    them to a Flask-readable format.
    """
    for msg in consumer:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + msg.value + b'\r\n\r\n') # Changed encoding so this no longer works


@app.route('/', methods=['GET'])
def home():
    return "Hello from flask"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)