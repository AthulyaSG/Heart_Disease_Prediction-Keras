# Heart_Disease_Prediction
Heart disease is one of the leading causes of death worldwide. The timely identification and accurate forecasting of cardiovascular disease are important to reduce death rates. Given that manual approaches are tedious and time-consuming, the application of neural networks to predict heart illness is explored. Neural network architecture is suggested, using various activation functions such as rectified linear units (ReLu), sigmoid, tanh, and softmax. Factor analysis is performed in order to reduce the dimensionality of the variables, where age, sex, chest pain, and resting blood pressure are chosen variables based on factor analysis. In this study, the neural network using 2 hidden layers with neurons 5 and a hidden layer with 3 neurons with relu activation function and sigmoid for output layer achieved the highest accuracy of 91.80%, whereas different numbers of hidden layers, i.e., 1 hidden layer with 7 neurons, 2 hidden layers with 5 neurons, and 2 hidden layers with 3 neurons with relu activation function and sigmoid for output layer achieves 47.54%; 2 hidden layers with relu function achieves 77.04%; and different activation functions of tanh for input and hidden layers and softmax for output has accuracy of 78.68%.

Cardiovascular diseases accounted for 17.9 million deaths worldwide in the year 2016, in which heart disease is the main condition. In order to prevent and predict heart defects, contributing factors should be understood. Angiography is expensive and tedious, so automated systems are being developed. The dataset used is obtained from the UCI repository with 303 rows and 14 variables, and the details on variables are given in Table 1.

![image](https://github.com/user-attachments/assets/3204dbf1-ba6a-49c5-aa80-9ab3465c9fa0)

Table 1: Variables table. From the website https://archive.ics.uci.edu/dataset/45/heart+disease.

Inspired by the human nervous network, the Artificial Neural Network (ANN) model replicates the learning process of the brain inside the system. Information is processed by the ANN through a network of interconnected neurons connected by communication links. There are three layers: an input layer receives information, hidden layers perform calculations, and output layer contains the result (Figure 1) (Lim et al., 2021). Activation functions introduce non-linearity into the neural network to learn complex patterns. Optimizers find how the weights are updated during training to minimize the error. Epoch is one pass through the training dataset.

![image](https://github.com/user-attachments/assets/8981db39-069f-412e-9c6a-025c40049d95)

Figure 1: Feedforward ANN. From Lim et al. (2021).

