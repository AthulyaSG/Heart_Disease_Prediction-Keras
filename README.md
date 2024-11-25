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

### Cap values using IQR

![image](https://github.com/user-attachments/assets/114e0b01-8287-41fd-a0c4-c0d7c65b3206)

Figure 8: The box plots after the removal of outliers.

Values are capped using IQR to limit the extreme values in the dataset in order to remove outliers, where values above the maximum are adjusted with the upper limit and values below the minimum are aligned with the lower limit.

## Exploratory data analysis and conclusion

![image](https://github.com/user-attachments/assets/2f1c2e79-ac94-41ef-bfda-a70ad9eb6f59)

![image](https://github.com/user-attachments/assets/24d3e37f-b598-43ca-9aac-5cd9d6aab356)

Figure 9: A Python code snippet displaying the summary of the dataset.

By describing the dataset, it is easy to understand the distribution of data. Count indicates the number of missing values. Minimum and maximum give the range of the data. Standard deviation and interquartile range denote the spread of the data. Percentiles are shown as 25%, 50%, and 75%.

### Histograms for continuous variables

![image](https://github.com/user-attachments/assets/26073670-d970-4d68-80d9-c436f69553a8)

Figure 10: Histograms for continuous variables.

Histograms for continuous values (age, resting blood pressure, cholesterol, maximum heart rate, and ST depression) are plotted to find the distribution (i.e., frequency), central tendency, and dispersion (spread) of the data. From Figure 10, it is noted that the majority of people belong to the age group of 57-60 years; blood pressure is concentrated from 125 to 135; cholesterol levels range between 240 and 260; maximum heart rates are observed at the range of 155-160; and ST depression is approximately 0-0.3.

### Pie charts for categorical values

![image](https://github.com/user-attachments/assets/39910808-b8c8-4e49-82f2-8a9996af23e7)

Figure 11: Pie charts for categorical values.

Pie charts are used to find the proportion of categorical values, where a slice indicates each category, and the entire data is represented by the whole pie. In Figure 11, 68% corresponds to category 1 for sex; 47.5% represents category 4 for chest pain; 67.3% is indicated in category 0 for exercise-induced angina; 46.9% is allocated to category 1 for slope; and 54.1% is attributed to category 0 for heart disease.


### Convert target multiclass to binary classification

![image](https://github.com/user-attachments/assets/08a688ea-7424-43d2-9eda-074d163fef92)

Figure 12: Pie chart for target class after conversion of multiclass to binary classification.

The target variable has multiple categories (0, 1, 2, 3, and 4); to fit into the model, the multiclass target value is converted to binary classification in heart disease.

### Grouped bar charts

![image](https://github.com/user-attachments/assets/91c9a127-a4f5-4e59-885b-9f26b4862791)

Figure 13: Grouped bar charts for categorical variables against heart disease.

Grouped bar charts are used to compare categorical values, where each group is represented by category. In chest pain, category 4 has individuals with the highest number of heart disease (>100) and category 3 has the highest number of people who are without heart disease. In gender class, male (>100) has the highest number of heart diseases. In the exercise-induced angina section, there seems to be no correlation between heart disease and exercise-induced angina as the highest number of people (around 140) are without heart illness. In slope, the number of people with heart disease (>100) in slope 2 is high.

### Heatmap

![image](https://github.com/user-attachments/assets/ece01906-de41-4fc0-9b76-30f74fa3e1ab)

Figure 14: Heatmaps.

Heatmaps are used to identify patterns and trends by colors and visualize relationships by positive and negative correlation and to determine the correlation between variables. A value near -1 indicates a strong negative correlation, 0 means a weak correlation, and near 1 is a strong positive correlation. There seems to be no positive or negative strong correlation between variables in the heatmap.

### Pair plots

![image](https://github.com/user-attachments/assets/ec31e547-da70-4b9f-ac70-87a76d6cdce2)

Figure 15: Pair plot

Scatter plots illustrate the relationship between variables by identifying patterns, trends, or correlations, utilizing continuous variables. When examining variables namely age, resting blood pressure, cholesterol, maximum heart rate, and ST depression against heart disease, there are no obvious trends, indicating any direct correlations. As the outliers are already handled using cap, there are no obvious outlie

## Factor Analysis

Factor analysis reduces dimensionality, where complex data into a small set of factors, identifying latent factors that explain correlations or patterns between variables.

### Normalize the data

![image](https://github.com/user-attachments/assets/a7164062-87a1-4150-b426-aa1bb9d2c8b1)

Figure 16. Normalize the values.

StandardScaler() is used to standardize the data to make all the values in the same way.

### Bartlett’s test and KMO

![image](https://github.com/user-attachments/assets/ebe5b71b-8b34-4c49-ae57-1116e557ea99)

Figure 17: A Python script of Bartlett’s test and output.

Bartlett’s test checks whether the variables are not related by assessing if the correlation matrix is like the identity matrix. A higher Chi-square value of 680.24 provides a larger discrepancy. A small P-value of 6.624 indicates that the null hypothesis is rejected, implying that there is a pattern in the relationship between variables. Thus, it concludes that the factor analysis provides meaningful relationships between the variables.

![image](https://github.com/user-attachments/assets/eea94c73-fb72-45f8-ae36-268847d86359)

The Kaiser-Meyer-Olkin (KMO) value of 0.6949 suggests the adequacy of data, with values near 1 suggesting better suitability.

### Determine the number of factors

![image](https://github.com/user-attachments/assets/17ae42ab-b20b-41f2-8e4a-37375ca017eb)

![image](https://github.com/user-attachments/assets/691a1f47-4f9d-448a-9e87-9bc93730b05e)

Figure 18: Eigenvalues to identify the factors.

Eigenvalues provide the variance of each variable. These are ordered in descending order to determine the number of factors, where eigenvalues greater than 1 are retained, and it can be visualized using a scree plot. Hence from Figures 18 and 19, it is evident that age, sex, chest pain, and resting blood pressure are 4 important factors taken into consideration.

![image](https://github.com/user-attachments/assets/ff8927a7-2ba6-4978-9e59-4fbf148329e2)

Figure 19: Scree plot

### Display top values

![image](https://github.com/user-attachments/assets/240a7b21-b850-4b18-8d47-3f573412814b)

Figure 20: Top values.

Top values are provided to identify extreme data points, where 5 top values are represented from each column.

## Prediction Task

In ANNs, activation functions, optimizers, and epochs are important components that influence the training process. ReLu is used for hidden layers; sigmoid, tanh, and softmax are used for output layers. The Adam optimizer is a learning rate optimization algorithm; 30 epochs with batch size 5 are used. The first four variables (age, sex, chest pain, and resting blood pressure) are chosen according to the factor analysis.

### Build the neural network using Keras after facto

![image](https://github.com/user-attachments/assets/d5f21694-30ff-492a-9d23-8445e3f6ff3d)

Figure 21. Build the neural network using Keras after factor analysis.

To build the neural network model, 3 hidden layers are used: 2 hidden layers with 5 neurons and 1 hidden layer with 3 neurons. The activation function used for input and hidden layers is relu and for output is sigmoid. The optimizer used is ADAM; 30 epochs are used; and binary cross-entropy is used as a loss function.

### Training loss

![image](https://github.com/user-attachments/assets/a425aefb-0ca6-4e7c-80ad-3aa11b388d6c)

Figure 22: Training loss

Loss is calculated on the training data during each epoch, where the epoch represents the number of iterations in training. Validation loss monitors the model’s performance and detects overfitting. A decrease in training loss indicates that the training set of data is what the model is learning. As loss decreases, it indicates that the model is learning effectively.

### Training accuracy

![image](https://github.com/user-attachments/assets/6f2f13c5-fa56-4e1d-978e-5ad2bf84180e)

Figure 23: Training accuracy.

Accuracy indicates the proportion of correctly identified instances in training data. Training accuracy monitors how the model is learning and how it reacts to hidden data. An increase in training accuracy indicates that the model is improving in the training data.

### Performance matrix

![image](https://github.com/user-attachments/assets/851f0573-fc88-4992-beb3-08fd6dba8031)

Figure 24: Performance matrix.

The accuracy of the model with factor analysis is 91.80%. The scores for precision, F1-score, and recall are 96%, 90%, and 85%, respectively.

### Parameters: Different numbers of hidden layers

![image](https://github.com/user-attachments/assets/1ae23ff2-4107-43a4-95ae-f00a77f7b741)

Figure 25: With different hidden layers.

In this model, 5 hidden layers are used: 1 hidden layer with 7 neurons, 2 hidden layers with 5 neurons, and 2 hidden layers with 3 neurons. The activation function used for input and hidden layers is relu and for output is sigmoid. The optimizer used is ADAM; 30 epochs are used with batch size 5.

### Performance matrix of different sizes of hidden layers

![image](https://github.com/user-attachments/assets/57564242-47b3-4d64-ba64-12ac75ca430c)

Figure 26: Performance matrix of different sizes of hidden layers.

The accuracy of using different sizes of hidden layers is 47.54%.

### Parameters: Reduced hidden layers

![image](https://github.com/user-attachments/assets/d6c222f7-8f1b-4250-b6fd-b135725a940e)

Figure 27: With reduced hidden layers.

As the previous model with 5 hidden layers provides only 47.5% accuracy, in this model, 2 hidden layers with 3 neurons each are used with relu, along with an input layer of size 3 with relu and an output layer of size 1 with sigmoid function. The activation function used for input and hidden layers is relu and for output is sigmoid. The optimizer used is ADAM; 30 epochs are used with batch size 5.

### Performance matrix of different sizes of hidden layers

![image](https://github.com/user-attachments/assets/51d8b0ab-7012-4e10-8ced-b8538d3c4ad1)

Figure 28: Performance matrix of different sizes of hidden layers.

The accuracy of using different sizes of hidden layers than the previous model is 77.04%. The scores of precision, recall, and F1-score are as follows: 76%, 81%, and 79%, respectively, outperforming the 5 hidden layers model.

### Different activation functions

![image](https://github.com/user-attachments/assets/660a6a6a-cc2a-4b60-aee5-f465e49546d9)

Figure 29: Different activation functions.

In this model, different activation functions are used in the layers, in which for input and 3 hidden layers tanh function is used, whereas softmax is used for the output layer.

### Performance matrix of different activation functions

![image](https://github.com/user-attachments/assets/eaebbdd5-102e-4af5-b7f0-98780fdefb3c)

Figure 30: Performance matrix of different activation functions.

The accuracy of the model for different activation functions is 78.68%. The scores for F1-score, recall, and precision are 77%, 84%, and 81%, respectively.
