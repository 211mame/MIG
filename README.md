# MIG
1.Concept
What I propose is a musical instrument generating application using machine learning .
This application, which I aim to develop, takes audio signal data as input and outputs 3D data of a wind instrument capable of producing the tone of the input data and I hope that the completion of this application will create an instrument with a unique tone. Also, #The reason for selecting a wind instrument instead of a percussion instrument or a string instrument is that if it is not a wind instrument, it is expected that the amount of program learning will be large. This will be described later.

2.Some learning methods
There are several ways and stages to build this application using machine learning.
I will explain the outline of some of the learning methods I'm thinking about at the moment.

Draft.1
First of all, The scale data taken over the specific instrument 3D model data such as .mesh file and the 12 sound technique that the instrument can emit is given as learning data. But we have to learn that if you plug any tone holes in the instrument program will hear sounds of such height and timbre. 
Therefore, it is necessary to give 3D model data that can be used to identify sound holes that are blocked each time one sound is given to the program as learning data. 
Although this method may be stable, there is very little 3D model data for instruments on the net, much less 3D model data blocking the tone holes corresponding to each scale. In other words, there is a high possibility of having to create 3D model data from scratch, and it is expected that the amount of work will be enormous. In order to reduce this huge amount of work, it is expected that measures will be taken such as automatic creation of 3D model data using GAN(Generative Adversarial Networks
) and narrowing the sound range to be input as learning data.

Draft.2
Draft.2 is a method using a large amount of image data. Compared with Draft.1, the difference is that 3D data is replaced with image data (photograph) and pass through multiple 2D model data before outputting as final 3D model data.
Naturally, there is no multilateral information in 2D model data, but that is supplemented by the quantity. Humans have to get the actual instrument, and for one sound we have to shoot from every angle and pass that data to the program. The advantage of this draft is that it is not necessary to make 3D model data as input, but it is necessary to construct a separate program to convert the 2D model data output by the program into a large amount of work of shooting and a program. There are problems such as difficulty in reproducing the hollow space inside the wind instrument.

Draft.3
