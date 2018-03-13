---
layout: post
title: JavaLand Real-Time Image Classification Demo
author: Nikolay Kostadinov
categories: [java, javaland, neural networks, python, keras, tensorflow, artificial intelligence, machine learning, imagenet, inception, classification]
---

This post could be very well considered a sequel to my talk at the JavaLand conference on 14th March 2018. It was my goal to 


# The Real-Time Image Classification Application

![png](/assets/images/real_time_classification_three_layers.png)

The real-time image classification application consists of three layers, which are loosely coupled through event-driven asynchronous communication. Fulfilling a different purpose, each of this three layers is implemented in a different programming language. In the following sections, you will find a brief description of each of these three layers as well as some explanations for the most interesting parts of the code. 

# Source Code
The source code is divided into two GitHub repositories: The [generic-realtime-image-classification-webapp](https://github.com/n-kostadinov/generic-realtime-image-classification-webapp) repo contains a Spring Boot application with a minimal JavaScript frontend, packaged as maven project. The [generic-realtime-image-classification-webapp](https://github.com/n-kostadinov/generic-realtime-image-classification-webapp) repo contains the Apache Kafka configuration and two image classification services that are further discussed in the following sections.

# Minimal Frontend Logic in JavaScript

![png](/assets/images/real_time_classification_gui.png)

As you can see from the image above, the frontend logic that is executed by your browser is written in JavaScript. You will find the source under: generic-realtime-image-classification-webapp/src/main/resources/static/assets/app.js. After loading the web page your machine's web camera will start capturing images. Using Google's [ImageCapture](https://developers.google.com/web/updates/2016/12/imagecapture) library this could be achieved by only a few lines of code. Caution: while [ImageCapture](https://developers.google.com/web/updates/2016/12/imagecapture) is natively supported by Google Chrome, it might not work in other browsers. ImageCapture is easiliy initialized:

```javascript
function initImageCapture(){
    navigator.mediaDevices.getUserMedia({video: true})
      .then(mediaStream => {
          
        const track = mediaStream.getVideoTracks()[0];
        imageCapture = new ImageCapture(track);
      }).catch(error => console.log(error));
}
```

The image is drawn into a HTML 5 Canvas and then sent to the backend given that the [STOMP](http://jmesnil.net/stomp-websocket/doc/) client is initialized. 

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

Clicking the "Connect" button on the minimalistic GUI will initialize the [STOMP](http://jmesnil.net/stomp-websocket/doc/) client:

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

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
