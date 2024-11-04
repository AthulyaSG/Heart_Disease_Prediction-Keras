# Heart_Disease_Prediction
Heart disease is one of the leading causes of death worldwide. The timely identification and accurate forecasting of cardiovascular disease are important to reduce death rates. Given that manual approaches are tedious and time-consuming, the application of neural networks to predict heart illness is explored. Neural network architecture is suggested, using various activation functions such as rectified linear units (ReLu), sigmoid, tanh, and softmax. Factor analysis is performed in order to reduce the dimensionality of the variables, where age, sex, chest pain, and resting blood pressure are chosen variables based on factor analysis. In this study, the neural network using 2 hidden layers with neurons 5 and a hidden layer with 3 neurons with relu activation function and sigmoid for output layer achieved the highest accuracy of 91.80%, whereas different numbers of hidden layers, i.e., 1 hidden layer with 7 neurons, 2 hidden layers with 5 neurons, and 2 hidden layers with 3 neurons with relu activation function and sigmoid for output layer achieves 47.54%; 2 hidden layers with relu function achieves 77.04%; and different activation functions of tanh for input and hidden layers and softmax for output has accuracy of 78.68%.

Cardiovascular diseases accounted for 17.9 million deaths worldwide in the year 2016, in which heart disease is the main condition. In order to prevent and predict heart defects, contributing factors should be understood. Angiography is expensive and tedious, so automated systems are being developed. The dataset used is obtained from the UCI repository with 303 rows and 14 variables, and the details on variables are given in Table 1.

![image](https://github.com/user-attachments/assets/3204dbf1-ba6a-49c5-aa80-9ab3465c9fa0)

Table 1: Variables table. From the website https://archive.ics.uci.edu/dataset/45/heart+disease.

Inspired by the human nervous network, the Artificial Neural Network (ANN) model replicates the learning process of the brain inside the system. Information is processed by the ANN through a network of interconnected neurons connected by communication links. There are three layers: an input layer receives information, hidden layers perform calculations, and output layer contains the result (Figure 1) (Lim et al., 2021). Activation functions introduce non-linearity into the neural network to learn complex patterns. Optimizers find how the weights are updated during training to minimize the error. Epoch is one pass through the training dataset.

![image](https://github.com/user-attachments/assets/8981db39-069f-412e-9c6a-025c40049d95)

Figure 1: Feedforward ANN. From Lim et al. (2021).

ANNs are used in the medical field for prediction tasks, as they are good at understanding complicated data. The ability of the networks to identify the hidden patterns in data is fascinating; ANNs improve the performance of forecasting, treatment, and treatment outcomes, helping doctors to make better decisions about diagnosis and treatment plans.

## Data types of variables

### Size of the data

![image](https://github.com/user-attachments/assets/8fbf4aab-56d5-4ae9-9c16-abc4b43ad47a)

![image](https://github.com/user-attachments/assets/59868679-9940-447a-8b92-f2c16c575ddc)

Figure 2: A Python code snippet showing the size of the data before and after providing column names.

Figure 2 indicates that the size of the data is 4228 and 4242 before and after providing column names, respectively.



### No. of columns and rows in the data

![image](https://github.com/user-attachments/assets/13c180c8-659e-4ab3-82ee-6f14ceff24f9)

Figure 3: A Python code snippet showing the number of observations and variables in the data.

As shown in Figure 3, there are 303 rows and 14 columns in the data.

### Renamed column names

![image](https://github.com/user-attachments/assets/6bf15ee3-595a-4409-b227-cb34ea000bd9)

Figure 4: A Python code snippet showing the renamed column names.

The column names have been renamed to provide meaningful labels to get the context of the data used.

### Data type of variables

![image](https://github.com/user-attachments/assets/7f6faa49-03af-4fff-a857-c97a6c371198)

![image](https://github.com/user-attachments/assets/dcc8d44d-9f19-4ec8-9262-e9f48aeb9664)

Figure 5: A Python code snippet showing the data types of columns and changing data types.

The column data types are displayed and changed data type ‘object’ to ‘float’ for column names ‘major vessels’ and ‘thal’.


## Data preprocessing

Handling missing values

![image](https://github.com/user-attachments/assets/f543955e-a9ab-4bfc-8369-af039b6ed5f1)

![image](https://github.com/user-attachments/assets/5999dccb-c1cb-4bd6-bd02-639b13e9797c)

![image](https://github.com/user-attachments/assets/363704ca-745d-4c99-b5ad-d5d33f462663)

Figure 6: A Python code snippet displaying null values and how it is handled.

Missing values are identified. Then, the symbols have been replaced by a mode function based on frequency (i.e., more number of occurrences).

### Outliers detection

![image](https://github.com/user-attachments/assets/0ec0df45-8d80-4c77-9654-ac7ea894f229)

Figure 7: The box plots to detect outliers.

Box plots are used to find outliers in the continuous variables. With box plots, minimum, maximum, 25th percentile, 50th percentile, and 75th percentile, and median can be determined. Values beyond maximum and minimum values are identified as outliers.

