---
layout: post
title: Kafka Micro Service, Part 2 of Cat vs Dog Real-Time Classification Series
author: Nikolay Kostadinov
categories: [python, catdog, artificial intelligence, machine learning, neural networks, convolutional neural network, GoogleLeNet, Inception, xgboost, ridgeregression, sklearn, tensorflow, image classification, imagenet, apache kafka, real-time]
---
This post is the second of a series of three. The goal is to embed a neural network into a real time web application for image classification. In this second part, I will put the machine learning model build in part one into use, by making it available through Apache Kafka - an open sources real-time event bus, widely adopted by the Big Data industry.

## Micro service for image prediction

In the first post of this series of three, I trained a second level classifier to be used on top of [Google's InceptionV3](https://arxiv.org/abs/1512.00567). In this second post I will embed the small model stack into a micro service that can be used for real-time image classification. The micro service will be both an event consumer and an event producer. Hence, it will listen for classification request events that contain the image to be classified and will respond by sending events that contain the classification label for the given image. Other than a simple REST service,[Apache Kafka](https://kafka.apache.org/) allows for the asynchronous communication between components. In a more complex setup, one may imagine that an event is processed by multiple components each containing a different stack of models. All responses are then aggregated and a decision is made based on the information gathered.

Let's start by importing all python dependencies that are necessary for the micro service to run. The [python client for kafka](https://github.com/dpkp/kafka-python) is fairly easy to install if you are running Ubuntu x64. All I had to run was " pip install kafka-python". 


```python
import tensorflow as tf
import xgboost
import pickle
from kafka import KafkaConsumer, KafkaProducer
from inception_v3 import inception_v3
from inception_v3 import inception_v3_arg_scope
import os
import inception_preprocessing
import numpy as np
import json
import base64
```

In the previous post of this series, I trained the second level classifier and stored it on the filesystem. It is a boosted trees classifier with the xgboost library. You can also find the classifier in the github repo, as it is actually very small, it takes only 122.2 kB on my file system. With just a few lines, we load the classifier and define a function that produces the label, which contains the cat/dog probabilities.


```python
with open('xgboost_model.p', 'rb') as handle:
    classifier = pickle.load(handle)
    
def predict_cat_dog(probs):
    cat_dog_prob = classifier.predict_proba(np.array(probs).reshape((1,-1)))[0]
    return 'Probabilities: cat {:.1%} dog {:.1%}'.format(cat_dog_prob[0], cat_dog_prob[1])
```

The first level classifier in the small stack of two is the InceptionV3 neural network that is already trained by Google. You should run the following lines of code and download InceptionV3 if you skipped the first part of this series.


```python
# DOWNLOAD DATASET 
from urllib.request import urlretrieve
import os
from tqdm import tqdm
import tarfile

inceptionv3_archive = os.path.join('model', 'inception_v3_2016_08_28.tar.gz')

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not os.path.isdir('model'):
    # create directory to store model
    os.mkdir('model')
    # download the model
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='InceptionV3') as pbar:
        urlretrieve(
            # I hope this url stays there
            'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
            inceptionv3_archive,
            pbar.hook)

    with tarfile.open(inceptionv3_archive) as tar:
        tar.extractall('model')
        tar.close()
```

At this point, you are all set to run the micro service. The InceptionV3 model is loaded and the tensorflow session is initialized. The kakfka consumer is registered for the "catdogimage" topic. A kafka producer is also initialized. For the sake of simplicity, there is a single data transfer object (DTO) that is both received from and sent back to the event bus. The DTO has the following structure:
    
    DTO:
       - label, stores the cat/dog label, empty when receiving.)
       - url, base64 encoded url of the image, it is never processed by the service, but is needed by the web application
       - data, base64 encoded image, that has been previously converted into png (Portable Network Graphics)


```python
INCEPTION_OUTPUT_SIZE = 1001
IMAGE_SIZE = 299
CHANNELS = 3 # Red, Green, Blue
INCEPTION_MODEL_FILE = os.path.join('model','inception_v3.ckpt')

slim = tf.contrib.slim

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
      
image_raw = tf.placeholder(dtype=tf.string)
image_data = tf.image.decode_png(image_raw, channels=3)
image = inception_preprocessing.preprocess_image(
            image_data, IMAGE_SIZE, IMAGE_SIZE, is_training=False)

expanded_image  = tf.expand_dims(image, 0)
with slim.arg_scope(inception_v3_arg_scope()):
        logits, _ = inception_v3(expanded_image, num_classes=INCEPTION_OUTPUT_SIZE, is_training=False)

probabilities = tf.nn.softmax(logits)

init_fn = slim.assign_from_checkpoint_fn(
        INCEPTION_MODEL_FILE, slim.get_model_variables())
    
with tf.Session() as sess:
    init_fn(sess)
    consumer = KafkaConsumer('catdogimage', group_id='group1')
    producer = KafkaProducer(bootstrap_servers='localhost:9092')
    for message in consumer:
        dto = json.loads(message.value.decode()) # Data Transfer Object
        image_data = base64.b64decode(dto['data'])
        np_probabilities = sess.run([probabilities], feed_dict={image_raw:image_data})
        dto['label'] = predict_cat_dog(np_probabilities)
        dto['data'] = None # no need to send image back
        producer.send('catdoglabel', json.dumps(dto).encode())
        print('Prediction made.', dto['label'])
```

    Prediction made. Probabilities: cat 99.9% dog 0.1%
    Prediction made. Probabilities: cat 99.9% dog 0.1%
    Prediction made. Probabilities: cat 99.9% dog 0.1%
    Prediction made. Probabilities: cat 99.9% dog 0.1%
    Prediction made. Probabilities: cat 100.0% dog 0.0%


The cycle call "for message in consumer" is blocking and will wait for an event. The DTO is then created by parsing from the json content that is in the message. The image data is decoded from base64 and feeded to the InceptionV3 neural network. The neural network produces the probabilities vector (with size 1001). Xgboost is used through the function defined above to create the final label. The label is then set into the DTO. The image just processed is removed from the DTO, as there is no need to send it back. The kafka producer is invoked - it sends the event with the "catdoglabel" topic. That's it. You have a real-time prediction service waiting for requests. In the final post of this series of three, I will create a small web application with spring boot that utilizes this service and allows users to classify images in real time. As always you can checkout the whole git repo here: [catdog-realtime-classification-kafka-service](https://github.com/n-kostadinov/catdog-realtime-classification-kafka-service).
