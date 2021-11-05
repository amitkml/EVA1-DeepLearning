Architectural Basics:
---------------------
Network 1 :
-----------------

Layers,Maxpooling,1x1,3x3,Receptive field

Considerable number of layer should be used such that  there is a right amount of receptive field for the network to predict the output .
it like how much a person is able to see a object .
eg: prediting the type tree when you see a 100ft tall tree at 1 m distance is lower that when you are 20m 

1x1 enables you to capture distinct feature from the available input with less number of parameters.

3x3 is a normal convolution that enables you to perform a 5x5 or 7x7 
operation .

using 1x7(1x(kernel)) and 7x1((kernel)x1) instead of 7x7 would save the number of paramters .

Softmax,Leanring rate,Kernels

As you can see in network 1 a proper block wise structure can be used 
which enables you to make the flow of neurons in a right manner

But this model will be overfitting mostly beacuse most of the model requires large number of kernels and the flow should be structured 
to make the model to pass test set.

Network 2:
----------


Batch normalization,dropouts,Position of maxpooling,epochs,transition layers,position of transition layers.

Batch normalization enables you to faster(make the values in a particular range)  flow of parameters in the model and avoid vanishing gradient problems.it should not be placed before a prediciton layer.

Dropout enables to find new path in backpropagation which enables the network to learn new activation function and also to avoid overfitting.This is used instead of data augumentation .

You should avoid positioning max pooling at the posterior and anterior ends of the network .it should be used where there is a need for increasing the receptive field.
Mostly at the middle,upper middle or lower middle part of the network.

BN,dropouts,maxpooling should not be used before prediction layers .As these things enable a network to perform well.


Network 3:
-------------

Adam vs sgd,CLR,reduceon pleatu

You can use adam or sgd but you have to use it with perfect parameters thus it enables better network performance.

You can know whether a network is performing well or not in the first few epochs itself .

CLR- cyclic learning rate enables you to adjust the learning rates during mini batch updates .

reduce on pleatu - you can monitor a specific parameter like validation loss .when the validation loss becomes constant you can change the learning rate at that time .

Network 4:
---------

LR scheduler

Better learning rate scheduler will enable you to control the step rate in gradient descent .



