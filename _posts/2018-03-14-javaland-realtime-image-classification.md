---
layout: post
title: JavaLand Real-Time Image Classification Demo
author: Nikolay Kostadinov
categories: [java, javaland, neural networks, python, keras, tensorflow, artificial intelligence, machine learning, imagenet, inception, classification]
---

The real-time image classification application is constructed as a minimal example of integrating a Deep Learning service into Spring Boot application. This post could be very well considered a sequel to my talk at the JavaLand conference on 14th March 2018. The goal of this post is to guide you through the code and enable you installing, modifying and running the real-time image classification application that was demonstrated in my JavaLand talk.

# The Real-Time Image Classification Application

POST WILL BE UPDATED WITH MORE VIDEOS AND MORE INFORMATION SOON...

## How to install and run the Application?

On the video below you can observe the whole process of installing and running the Real-Time Image Classification Application on a freshly started Amazon AWS instance. If you are not interested in the installation process, you can skip it and only watch the classification part here: [JavaLand: Real-Time Image Classification Demo](https://youtu.be/zQsE1qojRRA?t=14m46s).  

<iframe width="560" height="315" src="https://www.youtube.com/embed/zQsE1qojRRA" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## How does a Neural Network work?

The following slides explain the mechanics behind a Neural Network by simplifying and summarizing the online lectures of Andrew Ng. Find the full lecture here: [4.3.1 Neural Networks - Examples and Intuitions I](https://www.youtube.com/watch?v=1PZdrgWak8Y) and [4.3.2 Neural Networks - Examples and Intuitions II](https://www.youtube.com/watch?v=uySg3nBbkYg)

<iframe src="//www.slideshare.net/slideshow/embed_code/key/1CM8xtdHEWlkpJ" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/NikolayKostadinov/how-does-a-neural-network-work" title="How does a Neural Network work?" target="_blank">How does a Neural Network work?</a> </strong> from <strong><a href="https://www.slideshare.net/NikolayKostadinov" target="_blank">Nikolay Kostadinov</a></strong> </div>

## Some thoughts on the Architecture

![png](/assets/images/real_time_classification_three_layers.png)

The real-time image classification application consists of three layers, which are loosely coupled through event-driven asynchronous communication. Fulfilling a different purpose, each of this three layers is implemented in a different programming language. In the following sections, you will find a brief description of each of these three layers as well as some explanations for the most interesting parts of the code. 

## Source Code
The source code is divided into two GitHub repositories: The [generic-realtime-image-classification-webapp](https://github.com/n-kostadinov/generic-realtime-image-classification-webapp) repo contains a Spring Boot application with a minimal JavaScript frontend, packaged as maven project. The [realtime-image-classification-kafka-service](https://github.com/n-kostadinov/realtime-image-classification-kafka-service) repo contains the Apache Kafka configuration and two image classification services that are further discussed in the following sections.

## Minimal Frontend Logic in JavaScript

![png](/assets/images/real_time_classification_gui.png)

As you can see from the image above, the frontend logic that is executed by your browser is written in JavaScript. You will find the source under generic-realtime-image-classification-webapp/src/main/resources/static/assets/app.js. After loading the web page your machine's web camera will start capturing images. Using Google's [ImageCapture](https://developers.google.com/web/updates/2016/12/imagecapture) library this could be achieved by only a few lines of code. Caution: while [ImageCapture](https://developers.google.com/web/updates/2016/12/imagecapture) is natively supported by Google Chrome, it might not work in other browsers. ImageCapture is easily initialized:

```javascript
function initImageCapture(){
    navigator.mediaDevices.getUserMedia({video: true})
      .then(mediaStream => {
          
        const track = mediaStream.getVideoTracks()[0];
        imageCapture = new ImageCapture(track);
      }).catch(error => console.log(error));
}
```

The image is drawn into a HTML5 Canvas and then sent to the backend given that the [STOMP](http://jmesnil.net/stomp-websocket/doc/) client is also started. 

```javascript
function updateCanvasAndSendImage() {
	
	if ( imageCapture ) {
		
		imageCapture.grabFrame()
		  .then(imageBitmap => {
		    drawCanvas(imageBitmap);
		    sendJPEGImage()
		  })
		  .catch(error => console.log(error));
	}
	
}

function sendJPEGImage(){
	
	if( stompClient ) {
		stompClient.send("/app/webcamimage", {}, mainCanvas.toDataURL("image/jpeg"));
	}
	
}
```

Clicking the "Connect" button on the minimalistic GUI will start the [STOMP](http://jmesnil.net/stomp-websocket/doc/) client:

```javascript
function connect() {

	var socket = new SockJS('/websocket');
	stompClient = Stomp.over(socket);
	stompClient.connect({}, function (frame) {
	    setConnected(true);
	    console.log('Connected: ' + frame);
	    stompClient.subscribe('/topic/realtimeclassification', function (message) {
	    	setLabelsAndProbabilities(message)
	    });
	});
  
}
```

As you can see from the above code, the [STOMP](http://jmesnil.net/stomp-websocket/doc/) client is subscribed to a topic and waits for the classification labels to arrive.

## The Spring Boot Backend

![png](/assets/images/real_time_classification_springboot.png)

A controller in the Spring Boot backend is subscribed to the WebSocket and ready to receive the image sent by your Chrome browser:

```java
@Controller
public class RealTimeClassificationController {

	private static final String IMAGE_PREFIX = "data:image/jpeg;base64,";
	
	@Autowired
	private KafkaMessageProducer kafkaMessageProducer;
	
	@Autowired
	private ImageConverter imageConverter;
	
    @MessageMapping("/webcamimage")
    public void hangleWebcamImage(String imageDataUrl) {
		
		if ( imageDataUrl.startsWith(IMAGE_PREFIX)){
			
			String originalImageBase64JPEG = imageDataUrl.substring(IMAGE_PREFIX.length());
			String resizedImageBase64PNG = imageConverter.toBase64EncodedPNG(originalImageBase64JPEG);
			kafkaMessageProducer.publish(resizedImageBase64PNG);
		
		}

    }
			
}
```
The controller has to process the JPEG image and convert it to PNG image with size 299x299 - the size that is expected by the neural network deployed as a Kafka service. A minimal Kafka Producer is responsible for publishing the image to the "classificationimage"-Topic:

```java
@Component
public class KafkaMessageProducer {
	
	private final Logger logger = LoggerFactory.getLogger(this.getClass());
	
	@Value("${kafka.address}")
	private String kafkaAddress;
	
	@Value("${kafka.topic.classificationimage}")
	private String kafkaImageTopic;
	
	private Producer<String, String> kafkaProducer;
	
	@PostConstruct
	private void initialize() {
		
		Properties properties = new Properties();
		properties.put("bootstrap.servers", kafkaAddress);
		properties.put("acks", "all");
		properties.put("retries", 0);
		properties.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
		properties.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

		
		kafkaProducer = new KafkaProducer<>(properties);
	}	
	
	public void publish(String base64EncodedJPEGImage) {
		logger.info(String.format(
				"Sending encoded image, broker address %s, topic %s, size %d",
				kafkaAddress, 
				kafkaImageTopic,
				base64EncodedJPEGImage.length()));
		
		this.kafkaProducer.send(new ProducerRecord<String, String>(kafkaImageTopic, base64EncodedJPEGImage));
	}
	
	@PreDestroy
	public void close() {
		this.kafkaProducer.close();
	}
	
}
```

At the same time, a Kafka Consumer is subscribed to the "classificationlabel"-Topic and is waiting for labels published by image classification service. As you can see from the code above, the labels are rerouted to the JavaScript frontend without being altered in any way.

To run the Spring Boot application, navigate to the [generic-realtime-image-classification-webapp](https://github.com/n-kostadinov/generic-realtime-image-classification-webapp) and run from Maven by typing "mvn spring-boot:run". However, you should start Apache Kafka before running the Spring Boot application, otherwise, the Spring Boot application won't find a Kafka broker to subscribe to.

## Starting Apache Kafka

Apache Kafka is very easy to run if you have [docker installed](https://docs.docker.com/install/). Navigate to the realtime-image-classification-kafka-service/kafka_docker_config and run "docker-compose -f docker-compose-single-broker.yml up". In my particular case, I'am actually running "sudo docker-compose rm && sudo docker-compose -f docker-compose-single-broker.yml up" that is also contained in easy_run_kafka.sh script. You can find the minimal Kafka configuration for this demo in docker-compose-single-broker.yml config file.

## The Image Classification Service 

![png](/assets/images/real_time_classification_service.png)

Written in Python, the actual image classification service is quite compact. With just a few lines of code the image is decoded, preprocessed, classified by InceptionV3 and the labels are published back to Kafka.

```python
## Kafka Service
consumer = KafkaConsumer('classificationimage', group_id='group1')
producer = KafkaProducer(bootstrap_servers='localhost:9092')
for message in consumer:
    
    # transform image
    image_data = base64.b64decode(message.value.decode())
    pil_image = Image.open(BytesIO(image_data))
    image_array = img_to_array(pil_image)
    image_batch = np.expand_dims(image_array, axis=0)
    processed_image = inception_v3.preprocess_input(image_batch.copy())
    
    # make predictions
    predictions = inception_model.predict(processed_image)
    
    # transform predictions to json
    raw_label = decode_predictions(predictions)
    label = LabelRecord(raw_label)
    label_json = label.toJSON()
    
    # send encoded label
    producer.send('classificationlabel', label_json.encode())

```

To run the image classification service you will need Python 3.5 and also quite a lot of python libraries installed on your computer. Yet again, docker provides a shortcut. Just follow the instructions from this [Kaggle blog post](http://blog.kaggle.com/2016/02/05/how-to-get-started-with-data-science-in-containers/). However, instead running "docker pull kaggle/python" you should navigate to the generic-realtime-image-classification-webapp/python_install directory and run "sudo docker build -t "kaggleandkafka/python:dockerfile" ." This way you will also install the apache kafka client for python in addition to all the machine learning dependencies on the kaggle dockerfile. To run the service you can execute the Real_Time_Classification_Kafka_Service.py script. However, I prefer running the service from the [Jupyter Notebook](http://jupyter.org/). In the [generic-realtime-image-classification-webapp](https://github.com/n-kostadinov/generic-realtime-image-classification-webapp) repo you will also find the service code in the Real_Time_Classification_Kafka_Service.ipynb.

## Transfer Learning

If you have made it so far to run the Kafka cluster, the Spring Boot application and the image classification service, you may want to continue with this a little more advanced example of transfer learning. Download the [dog images](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and unpack these in the [generic-realtime-image-classification-webapp](https://github.com/n-kostadinov/generic-realtime-image-classification-webapp) repo. Run the Train_Dog_Breed_Model.py script to train your own neural network that accepts the output of InceptionV3 as its input. Running this script takes more than an hour when running on my laptop. After the script has terminated you will see the newly created dogbreed_model.hdf5 file in the repo. This file contains the weights of the neural network. Now you can run Real_Time_Dog_Breed_Classification_Kafka_Service.py script to start the "Dog Breed" image classification service. 

## Need help?
if you need some help running the above code, feel free to contact me! Also make sure you check this page in the following few days, as the information in this post will be updated...
