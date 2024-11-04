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

