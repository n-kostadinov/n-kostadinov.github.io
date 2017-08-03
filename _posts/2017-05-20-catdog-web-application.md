---
layout: post
author: Nikolay Kostadinov
title: Cat vs Dog Real-time classification, Part 3 of Cat vs Dog Real-Time Classification Series
categories: [catdog, apache kafka, real-time, java, javascript, springboot, websockets, html, stomp, kafka, event pipeline]
---

This post is the second of a series of three. The goal is to embed a neural network into a real time web application for image classification. In this third part, I will go through a small, yet deployable application built with the Spring Framework. Furthermore, I will elaborate on how I used Apache Kafka to put it all together and embed the micro service introduced in part 2 in my client-server environment.

# Architecture Overview
In the first post of this series of three [Model Stack (Part 1)](http://machinememos.com/python/catdog/artificial%20intelligence/machine%20learning/neural%20networks/convolutional%20neural%20network/googlelenet/inception/xgboost/ridgeregression/sklearn/tensorflow/image%20classification/imagenet/apache%20kafka/real-time/2017/05/18/catdog-stacked-classification.html) I demonstrated how one could use the pre-trained model [InceptionV3](https://arxiv.org/abs/1512.00567) to create a small stack that can predict if there is a cat or a dog on an image with an accuracy of over 99%. In the second post [Kafka Micro Service](http://machinememos.com/python/catdog/artificial%20intelligence/machine%20learning/neural%20networks/convolutional%20neural%20network/googlelenet/inception/xgboost/ridgeregression/sklearn/tensorflow/image%20classification/imagenet/apache%20kafka/real-time/2017/05/19/catdog-kafka-microservice.html) the predictive power of the classification stack is encapsulated in a micro service that can be accessed through the Apache Kafka event bus. In this last post of this series I will leave the newly born realm of AI and machine learning realm to enter once again the already mature universe of software engineering for the web. However, I do not intend to go into great detail on how [Apache Kafka](https://kafka.apache.org/) or [Web Sockets](https://www.websocket.org/aboutwebsocket.html) work. Neither will I post the complete code here, as you can directly check it out from the git repo: [catdog-realtime-classification-webapp](https://github.com/n-kostadinov/catdog-realtime-classification-webapp), build it and run it within a few minutes. The goal here is to only give you the general idea how the micro service and the web application interact. 

## Apache Kafka
If you look at the official website of [Apache Kafka](https://kafka.apache.org/)  you will find the following statement: "Kafka is used for building real-time data pipelines and streaming apps. It is horizontally scalable, fault-tolerant, wicked fast, and runs in production in thousands of companies." The reason I chose Kafka for this demo is namely the "real-time data pipelines" capability. Horizontal scalability is a "must" in the domain of web and business applications.  There are several other promising technologies, e.g. Vert.x and Akka. Unfortunately the current version of [Vert.x (c.3.4.1)](http://vertx.io/) does not have a python client. [Akka](http://akka.io/downloads/) is native to the JVM and officially supports only Java and Scala. Next to the real-time data pipeline and very easy to handle programmatic interfaces for both java and python, there are numerous "plug & play" docker containers to choose from in GitHub. Hence, for my demo I created a fork of [kafka-docker by wurstmeister](https://github.com/wurstmeister/kafka-docker).  You can checkout my fork here: [kafka-docker](https://github.com/n-kostadinov/kafka-docker). As mentioned in the README.md one has to only install docker-compose and call "docker-compose -f docker-compose-single-broker.yml up" to get kafka up and running.

## UI with Web Sockets
The user interface of the web application is very simple. It is slightly modified version of the JavaScript client shipped with the Spring web sockets example:  [Using WebSocket to build an interactive web application](https://spring.io/guides/gs/messaging-stomp-websocket/). 

![png](/assets/images/catdogui.png)

The client side logic is kept to the minimum. There is no angular2, the only libraries used are jquery, sockjs and stomp. I can only recommend this very nice written [tutorial on sockets and stomp.](http://jmesnil.net/stomp-websocket/doc/) Anyway, the client binds to the "catdog-websocket" and subscribes to the "topic/catdog":

{% highlight javascript %}
function connect() {
    var socket = new SockJS('/catdog-websocket');
    stompClient = Stomp.over(socket);
    stompClient.connect({}, function (frame) {
        setConnected(true);
        console.log('Connected: ' + frame);
        stompClient.subscribe('/topic/catdog', function (message) {
            showCatDogDTO(message);
        });
    });
}
{% endhighlight %}

If a message is received the client handles its content and modifies the HTML layout by appending new rows to the table:
{% highlight javascript %}
function showCatDogDTO(unparsed_message) {
	
	time = get_time()
	preffix = "<tr><td>[" + time +"] "
	suffix = "</td></tr>"
	
	message = JSON.parse(unparsed_message.body)
	if (message.resolved) {
		$("##catdogboard").prepend(
	    preffix + "Image processed by Deep-Cat-Dog" + suffix +
		preffix + "Image as seen by the Neural Network</td></tr>" + suffix + 
		preffix + "<img (...) src=\"" + message.url + "\"/>" + suffix +
		preffix + "Label: " + message.label + suffix);
	} else {
		$("##catdogboard").prepend(preffix + message.content + suffix);
	}
}
{% endhighlight %}

## Kafka Message Producer

Whenever the user sends a message through the input form, a spring controller checks if it contains an URL to an image. If so the image is downloaded and converted to png. The KafkaMessageProducer is responsible for sending the message to Kafka's event bus. It's amazing how few lines of code are needed to create a Kafka Message Producer that does the job.

{% highlight java %}
@Component
public class KafkaMessageProducer {
	
	private final Logger logger = LoggerFactory.getLogger(this.getClass());

	private static final String BROKER_ADDRESS = "localhost:9092";
	private static final String KAFKA_IMAGE_TOPIC = "catdogimage";
	
	private Producer<String, String> kafkaProducer;
	
	@PostConstruct
	private void initialize() {
		Properties properties = new Properties();
		properties.put("bootstrap.servers", BROKER_ADDRESS);
		properties.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
		properties.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
		kafkaProducer = new KafkaProducer<>(properties);
	}	
	
	public void publish(CatDogKafkaDTO catDogEvent) {
		
		logger.info("Publish image with url " + catDogEvent.getUrl());
		String jsonValue;
		try {
			jsonValue = new ObjectMapper().writeValueAsString(catDogEvent);
		} catch (JsonProcessingException ex) {
			throw new IllegalStateException(ex);
		}
		this.kafkaProducer.send(new ProducerRecord<String, String>(KAFKA_IMAGE_TOPIC, jsonValue));
	}
	
	@PreDestroy
	public void close() {
		this.kafkaProducer.close();
	}
	
}
{% endhighlight %}

The publish call is non-blocking. The event is received on the other side and the image is classified with InceptionV3 and xgboost classifier trained in part one: [Cat vs Dog Real-time classification: Model Stack (Part 1)](http://machinememos.com/python/catdog/artificial%20intelligence/machine%20learning/neural%20networks/convolutional%20neural%20network/googlelenet/inception/xgboost/ridgeregression/sklearn/tensorflow/image%20classification/imagenet/apache%20kafka/real-time/2017/05/18/catdog-stacked-classification.html)

## Kafka Message Consumer

Eventually, the python script will send an event containing the classification message. For more details on how this script works, check out part 2: [Cat vs Dog Real-time classification: Kafka Micro Service (Part 2)](http://machinememos.com/python/catdog/artificial%20intelligence/machine%20learning/neural%20networks/convolutional%20neural%20network/googlelenet/inception/xgboost/ridgeregression/sklearn/tensorflow/image%20classification/imagenet/apache%20kafka/real-time/2017/05/19/catdog-kafka-microservice.html) The event will be received and handled by a spring the spring component bellow:

{% highlight java %}

	@PostConstruct
	public void initilize() {
		logger.info("initilize");
		kafkaConsumer = new KafkaConsumer<>(properties);
		kafkaConsumer.subscribe(Arrays.asList(KAFKA_LABEL_TOPIC));
	}

	@Scheduled(fixedRate = 100)
	public void consume() {
		
		ConsumerRecords<String, String> records = kafkaConsumer.poll(0L);
		Iterator<ConsumerRecord<String, String>> iterator = records.iterator();
		
		while(iterator.hasNext()){
			ConsumerRecord<String, String> consumerRecord = iterator.next();
			safelyProcessRecord(consumerRecord);
		}

	}

}
{% endhighlight %}

The consume method gets triggered every 100 milliseconds and polls for a Kafka message. If there is an event, the consumer has not handled yet, it gets processed and another event is send to the browser client. The showCatDogDTO javascript funciton shown above gets called and the data gets written on the HTML page. 

## The real-time cat / dog classifier

For the sake of simplicity I commented only a small part of the web application code. Be sure to check out the git repo, build it and run it: [catdog-realtime-classification-webapp](https://github.com/n-kostadinov/catdog-realtime-classification-webapp) Or you can watch the youtube video below to get a taste on how everything works and feels like:

<iframe width="640" height="360" src="https://www.youtube.com/embed/P1GdfLyjSek" frameborder="0" allowfullscreen></iframe>

As tech is further advancing it gets increasingly easier to build real-time smart applications. Data scientists can script micro services that are bound together with Apache Kafka or similar real-time data pipeline. This way one can skip or significantly reduce long production cycles, where prototypes written by data scientists are recoded in Java and directly embedded in monolithic business applications. Real-time pipelines give you so much flexibility, you can change scripts in production dynamically, and test these on a small sample of users at first. The future is surely going to be amazing.



