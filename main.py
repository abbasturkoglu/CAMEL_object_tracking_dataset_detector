import base64
import datetime
import pickle

import cv2
import numpy as np
import torch

from flask import Flask, Response
from kafka import KafkaConsumer, KafkaProducer
from PIL import Image
from detector.detector import get_transform, get_fasterrcnn_model


def from_base64(buf):
    buf_decode = base64.b64decode(buf)
    buf_arr = np.fromstring(buf_decode, dtype=np.uint8)
    return cv2.imdecode(buf_arr, cv2.IMREAD_UNCHANGED)


topic_in = "raw-video"
topic_out = "object-detections"

def main():

    model = get_fasterrcnn_model()
    model.eval()
    topic = "raw-video"
    transform = get_transform(False)

    consumer = KafkaConsumer(
        topic_in,
        bootstrap_servers=['kafka1:19091']
    )

    producer = KafkaProducer(bootstrap_servers=['kafka1:19091'])

    image_id = -1

    for msg in consumer:

        #img = decode(msg.value)
        img = from_base64(msg.value)
        image = Image.fromarray(img)
        image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)

        # img = Image.open('./data/sequence-1/img1/000608.jpg').convert("RGB")

        image_tensor = transform(image, {})
        image_tensor = image_tensor[0].unsqueeze(0).float()

        with torch.no_grad():
            predictions = model(image_tensor)

        for frame in zip(predictions):

            frame = frame[0]
            image_id += 1
            boxes = frame['boxes'].cpu().numpy()
            labels = frame['labels'].cpu().numpy().reshape(-1,1)
            scores = frame['scores'].cpu().numpy().reshape(-1,1)
            image_id_col = np.array([image_id]*scores.shape[0]).reshape(-1,1)
            x = np.array([-1]*scores.shape[0]).reshape(-1,1)
            y = np.array([-1]*scores.shape[0]).reshape(-1,1)
            z = np.array([-1]*scores.shape[0]).reshape(-1,1)

            # Convert bbox coords to tlwh from tlbr
            boxes[:,2] = boxes[:,2]-boxes[:,0]
            boxes[:,3] = boxes[:,3]-boxes[:,1]

            frame_results = np.hstack([image_id_col, labels, boxes, scores, x, y, z])

        # Convert to bytes and send to kafka
        # producer.send(topic, buffer.tobytes())
        buffer = pickle.dumps({'image_id':image_id , 'img':msg.value, 'frame_results':frame_results})
        producer.send(topic_out, base64.b64encode(buffer))


if __name__ == '__main__':
    main()
