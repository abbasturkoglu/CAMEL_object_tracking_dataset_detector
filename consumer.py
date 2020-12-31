import base64
import datetime
from kafka import KafkaConsumer



# Fire up the Kafka Consumer
topic = "raw-video"

consumer = KafkaConsumer(
    'raw-video',
    bootstrap_servers=['kafka1:19091']
)

print(consumer.topics())

#Here is where we recieve streamed images from the Kafka Server and convert
#them to a Flask-readable format.

for msg in consumer:
    print(msg.value)
