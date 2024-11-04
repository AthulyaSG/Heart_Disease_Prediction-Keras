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

# Literature Review

Using machine learning (ML) models, including XGBoost, Random Forest (RF), Decision Tree (DT), Logistic Regression (LR), Bagging, k-nearest neighbor (KNN), LightGBM, and Support Vector Machine (SVM), the paper by (Mamun et al., 2022) focused on developing models for prediction for the survival of patients with heart failure (HF). In this study, to balance the dataset, SMOTE, an oversampling approach, was employed. As this was a small dataset with only 299 rows and 13 variables, K-fold cross-validation was employed to achieve accurate results, where the LightGBM model had the highest accuracy (0.85) and area under the curve (AUC) (0.93). Smaller dataset was a constraint that was overcome by increasing the size of the dataset using data augmentation. In the future, this can be further enhanced by adding larger datasets to improve the accuracy of predictions and using deep learning (DL) algorithms.

The paper by (Sujatha & Mahalakshmi, 2020) emphasized the importance of ML in detecting and diagnosing heart illness, a major cause of death worldwide. DT, RF, SVM, LR, DT, and KNN were used in ML techniques with performance matrices such as Accuracy, AUC, F1-score, and Precision. For feature selection, the algorithm used was Boruta, in which 6 variables were taken into account. After feature selection, the implementation of supervised learning was done, where the RF classifier outperformed with an accuracy of 0.8351.

Similarly, Zhang & Zhou (2021) in their work used a small dataset with 299 records but with different 6 features and proposed a predictive model for HF using a Genetic Algorithm (GA)-optimized Extreme Learning Machine (ELM); the ELM is a fast and accurate method, where random input can cause instability, and is improved by GA, a reliable and thorough model. This study provided a balance between performance and efficiency by selecting the number of hidden layer nodes as 40 and the activation function as sigmoid with an accuracy of 0.786.

The authors (Li et al., 2023) proposed a vector to determine whether a value needed to be padded and was correct, which increased the size of the data and fixed missing values utilizing models, namely SVM, Multilayer Perceptron (MLP), Multi-head self-attention convolutional neural network (CNN), RF, KNN, LR, and Light Gradient Boosting Machine (LGB). Data imbalance was addressed by the focal loss function. For the training approach, 5-fold cross-validation prevented overtraining and inadequate training. Deep SHapley Additive exPlanations (SHAP) integrated DeepLIFT, assigning weights to input features during backpropagation, and SHAP, providing unique scores for individual input. Accuracy rates of deaths for different timings were given as follows: within 30 days, RF achieved 86.31%; within 365 days, LGB attained 89.12%; and within 180 days and after 365 days, DLS-MSM recorded 82.08% and 76.07% accuracy rates, respectively.

The goal of the article by (Sharma and Parmar, 2020) was to improve efficiency and accuracy in heart attack prediction algorithms, comprising RF, NB, Talos-optimized deep learning neural network (DNN), SVM, LR, and KNN with 14 features and 303 data. Data preprocessing was done by vectorization of features, indicator vector for missing values, and standardization. Talos optimizer served as a method for hyperparameter improvement along with the Keras library. The Prepare, Optimize, and Deploy process was used. Firstly, in the Prepare step, hyperparameter space was defined. Secondly, in the Optimize stage, finding the hyperparameters was automated; and in the Deploy procedure, deployment of local and remote models facilitated automation. Finally, after the above-mentioned procedures, 90.78% was the highest accuracy obtained by optimizing hyperparameters using Talos.

This study by (Andari et al., 2023) employed ML (Naïve Bayes (NB), LR, and KNN) and artificial neural networks (MLP) with backpropagation and was examined with different hyperparameter values to achieve improved accuracy for predicting a heart attack. In this study, various values of hyperparameters were utilized to assess the accuracy and achieve results. The highest accuracy was achieved at 83.051% by LR with 5 folds, whereas NB with 10 folds had the highest accuracy of 75.862%. In KNN, about 77.419% of the highest accuracy was attained at k = 8. The highest accuracy (0.8406) was obtained by MLP using backpropagation along with 100 epochs, 0.1 learning rate, 1 hidden layer, and 5 folds.

The study by (Mohan et al., 2021) aimed to apply 4 ML methods (NB, LR, KNN, and RF) to automate the prediction of heart disease instead of using angiography, which is costly, wherein LR outperformed other models with the highest accuracy of 90.2%. This study considered the following risk factors: controllable including nicotine usage, alcohol consumption, and inadequate exercise and uncontrollable namely age, sex, and family background. The dataset was split into training of the ML model on the training set and testing of the ML model on the testing set.

In order to reduce risk, accelerate treatment, and provide early detection, the paper seemed to determine the most accurate model to predict heart attacks, by employing ML algorithms. Digital signal processing was used for the cleaning of the UCI dataset. This study employed the following methods RF, DT, KNN, and LR to train the dataset and to test the models’ accuracy. In future work, DL algorithms can be implemented to outperform supervised learning techniques (Battula et al., 2021).

This article by (Narsimhulu et al., 2022) investigated the use of ML in the prediction of heart illness and discussed the problem of accuracy being dependent on training, which degraded further because of its dimensionality. Using the AI-driven Filter Based Feature Selection method, only important features were considered, and unnecessary features were removed in order to make better predictions. The findings showed that using Filter Based Feature Selection method improved the performance of the RF model where the accuracy was 95.08% compared to NN, SVM, DT, LR, NB, XGBoost, and KNN with accuracy of 80.33%, 82.97%, 82.97%, 86.26%, 86.25%, 87.25%, and 67.21%, respectively, thus indicating the need for good algorithm for the selection of features.

In general, the external examination involved examining physical results, detecting signs and symptoms, and exploring records, whereas the internal examination included surgery (Miranda et al., 2021). This study proposed a prediction model for early heart failure detection using LR and Stochastic Gradient Descent (SGD). Gradient descent was one of the ML techniques used to improve neural networks: Batch Gradient Descent guaranteed the lowest local and global scope sizes for non-convex and convex errors, respectively; SGD duplicated computations on a big dataset; and Mini-batch gradient descent designed for the training of the neural network. LR achieved the highest accuracy of 91.67% when compared to stochastic gradient descent with 80.00% accuracy.

Using ML models like NB, SVM, DT, KNN, and RF, the study by (Hamdaoui et al., 2020) suggested a clinical decision-making aid to forecast heart ailments. Data was preprocessed by imputing mode to the missing values and normalizing the values; then, it was split into training and test sets. Cross-validation was suggested as a way to avoid overperformance, which used a 10-fold approach for better evaluation and lowered errors (Figure 31). Although many models were used in this work, NB achieved the highest accuracy of 84.28% with the split data method and 82.17% with cross-validation.

![image](https://github.com/user-attachments/assets/4bdde854-87c6-4eaa-90f9-87d121a931d8)

Figure 31: Flow diagram of the work by (Hamdaoui et al., 2020).

This project aimed to use supervised ML for early forecasting of heart disease, which was done by the SVM, MLP, and KNN (Kumar et al., 2023). By employing metrics, Euclidean and Manhattan distances, KNN calculated the closeness of data of the model’s prior training data to identify cardiovascular disease (Figure 32). SVM classifier assigned based on distance by a hyperplane with training data. The Multi-layer Perceptron comprised numerous layers of perceptron and utilized for classifying and segmenting data from the medical field. The creation of a neural network using TensorFlow. Normalization was used as a data preprocessing technique. KNN achieved the highest accuracy of 91.8%.

![image](https://github.com/user-attachments/assets/4fd0d937-b378-475d-84d6-c0887bff6f57)

Figure 32: Block diagram from the paper by (Kumar et al., 2023).

This work by (Shaker et al., 2022) emphasized the importance of cardiovascular defects, highlighting the prevalence and difficulties associated with angiography, which is costly and unsafe; this was done by employing ML (KNN, DT, RF, SVM (polynomial, linear, sigmoid, and Gaussian kernels), NB, and Voting classifier) and DL (CNN, RNN, and Long Short-Term Memory (LSTM)) models (Figure 33). Data cleaning was done by the selection of features, where the Lasso algorithm was utilized, and PCA to find the importance of the datasets. RF classifier had reached an accuracy of 93%.

![image](https://github.com/user-attachments/assets/7647edc4-ed03-4e72-a6c6-57d654a1f6a4)

Figure 33: Flow diagram of the proposed method by (Shaker et al., 2022).

This work by (Lim et al., 2021) explored the usage of ANN in the diagnosis of cardiovascular defects in the Philippines. Data cleaning procedures included transformation into arrays and then divided into variables and output, followed by normalization for data reduction (Figure 34). The training of the ANN model was performed in the following order: input layer, with 13 neurons and without an activation function; then, 2 hidden layers with 25 neurons along with an activation function as a rectified linear unit (ReLu); and output layer with 1 neuron with a sigmoid activation function. The loss function calculated the metric in order to reduce training. ANN achieved high accuracy with a training accuracy of 89.13% and was further validated by k-fold cross-validation to achieve an accuracy of 87.63%.

![image](https://github.com/user-attachments/assets/23791113-1d56-4e15-8a85-a684633601af)

Figure 34: Proposed ANN framework by (Lim et al., 2021).

In this article by (Ramprakash et al., 2020), to evaluate the model’s effectiveness, ANN and DNN were employed, where DNN was characterized by many hidden layers and χ2-DNN was created to improve the accuracy of the classification. Neural networks, algorithmic sets identifying patterns and trends, comprising layers with activation functions. Forward propagation was formulated employing an output layer with a sigmoid activation function. Feature extraction and feature selection were used to retrieve valuable data and filter duplicate values, respectively. Grid search improved features till the best set was obtained. Layer 4 achieved the highest accuracy of 94% with 400 epochs. Future research may employ genetic algorithms to improve performance.

In the work by (Raihan et al., 2019), data cleaning was performed by trimming the values of mode, mean, and median. TensorFlow was utilized, where Keras served as the front end. The ANN comprised 14 neurons in the input layer; 13 neurons in a hidden layer; and 2 neurons in an output layer, with the combination of sigmoid activation function and 0.01 learning rate. The network weights were set as small by the backpropagation algorithm. The accuracy of ANN to predict the defect in the heart was 84.47%.

Hybrid Deep Neural Networks (HDNNs) employed hybrid CNN-LSTM to improve the prediction of diagnosis of heart failure, to automate learning, and to integrate results for decision-making (Reshan et al., 2023). Data was filtered by using Standard Scalar, deleting empty rows, changing multiclass to binary classification, and deleting outliers. Feature selection employed an Extra Tree Classifier with Gini, which identified important features affecting prediction performance (Figure 35). In ANN, the number of neurons used in input was similar to the first hidden layers and the second hidden layer had 5 neurons with ReLu as the activation function; the softmax activation function was used for outer layer. In CNN, kernels provided feature mappings using stride, where zero padded was used to increase input volume; the pooling layer reduced the dimensionality by performing max pooling; and the Fully Connected layer utilized information from the above-mentioned layers for the classification of data. LSTM contained forget gate, input gate, candidate gate, and output gate. While using the hybrid CNN-LSTM model, there were 30 layers, where the convolutional layer had 18 layers; pooling layers had 12; and 1 for each fully connected layer, LSTM, and output; and finally, a dropout rate of 20% occurred in the dropout layer. CNN-LSTM achieved the highest accuracy (97.75% and 98.86%) in both datasets ((1) Cleveland and (2) Switzerland + Cleveland + Statlog + Hungarian + Long Beach VA, respectively).

![image](https://github.com/user-attachments/assets/4060df7a-27c9-4f71-9e7f-076fde817de3)

Figure 35: Flowchart of the proposed system by (Reshan et al., 2023).

The fully connected structure of the suggested dense neural network was demonstrated by the interconnection of all nodes, where each node was connected to other nodes (Singh & Jain, 2022). Preprocessing of data was done by deleting or converting null rows accordingly and transforming categorical values into numerals (Figure 36). The cleaned data was split into training and test sets. In the dense neural networks with 100 epochs, 3 neurons were selected for input layers, 4 in hidden layers, and 2 in output layers. The dense neural network achieved an accuracy of 96.09%.

![image](https://github.com/user-attachments/assets/12641e82-2a07-4bf4-867e-8511b8f47455)

Figure 36: Flow chart of the proposed method by (Singh & Jain, 2022).

References
Andari, B., Owayjan, M., Haidar, G. A., & Achkar, R. (2023). Heart Failure Prediction Using Machine Learning and Artificial Neural Networks. 2023 Seventh International Conference on Advances in Biomedical Engineering (ICABME), 257–261. https://doi.org/10.1109/ICABME59496.2023.10293102
Battula, K., Durgadinesh, R., Suryapratap, K., & Vinaykumar, G. (2021). Use of Machine Learning Techniques in the Prediction of Heart Disease. 2021 International Conference on Electrical, Computer, Communications and Mechatronics Engineering (ICECCME), 1–5. https://doi.org/10.1109/ICECCME52200.2021.9591026
Hamdaoui, H. E., Boujraf, S., Chaoui, N. E. H., & Maaroufi, M. (2020). A Clinical support system for Prediction of Heart Disease using Machine Learning Techniques. 2020 5th International Conference on Advanced Technologies for Signal and Image Processing (ATSIP), 1–5. https://doi.org/10.1109/ATSIP49331.2020.9231760
Kumar, K. P., Rohini, V., Yadla, J., & VNRaju, J. (2023). A Comparison of Supervised Learning Algorithms to Prediction Heart Disease. 2023 International Conference on Artificial Intelligence and Knowledge Discovery in Concurrent Engineering (ICECONF), 1–5. https://doi.org/10.1109/ICECONF57129.2023.10084035
Li, D., Fu, J., Zhao, J., Qin, J., & Zhang, L. (2023). A deep learning system for heart failure mortality prediction. PLOS ONE, 18(2), e0276835. https://doi.org/10.1371/journal.pone.0276835
Lim, R. M., Munsayac, F. E. T., Bugtai, N. T., & Baldovino, R. G. (2021). A Predictive Tool for Heart Disease Diagnosis using Artificial Neural Network. 2021 IEEE 13th International Conference Humanoid, Nanotechnology, Information Technology, Communication and Control, Environment, and Management (HNICEM), 1–4. https://doi.org/10.1109/HNICEM54116.2021.9731858
M. Tech. Scholar,Dept. of CSE & IT, MITS Gwalior, Gwalior, Madhya Pradesh, India., Sharma*, S., Parmar, M., & Assistant Professor MITS,Dept. of CSE & IT, MITS Gwalior, Gwalior, Madhya Pradesh, India. (2020). Heart Diseases Prediction using Deep Learning Neural Network Model. International Journal of Innovative Technology and Exploring Engineering, 9(3), 2244–2248. https://doi.org/10.35940/ijitee.C9009.019320
Mamun, M., Farjana, A., Mamun, M. A., Ahammed, M. S., & Rahman, M. M. (2022). Heart failure survival prediction using machine learning algorithm: Am I safe from heart failure? 2022 IEEE World AI IoT Congress (AIIoT), 194–200. https://doi.org/10.1109/AIIoT54504.2022.9817303
Miranda, E., Bhatti, F. M., Aryuni, M., & Bernando, C. (2021). Intelligent Computational Model for Early Heart Disease Prediction using Logistic Regression and Stochastic Gradient Descent (A Preliminary Study). 2021 1st International Conference on Computer Science and Artificial Intelligence (ICCSAI), 11–16. https://doi.org/10.1109/ICCSAI53272.2021.9609724
Mohan, N., Jain, V., & Agrawal, G. (2021). Heart Disease Prediction Using Supervised Machine Learning Algorithms. 2021 5th International Conference on Information Systems and Computer Networks (ISCON), 1–3. https://doi.org/10.1109/ISCON52037.2021.9702314
Narsimhulu, K., Ramchander, N. S., & Swathi, A. (2022). An AI Enabled Framework with Feature Selection for Efficient Heart Disease Prediction. 2022 5th International Conference on Contemporary Computing and Informatics (IC3I), 1468–1473. https://doi.org/10.1109/IC3I56241.2022.10073155
Raihan, M., Mandal, P. K., Islam, M. M., Hossain, T., Ghosh, P., Shaj, S. A., Anik, A., Chowdhury, M. R., Mondal, S., & More, A. (2019). Risk Prediction of Ischemic Heart Disease Using Artificial Neural Network. 2019 International Conference on Electrical, Computer and Communication Engineering (ECCE), 1–5. https://doi.org/10.1109/ECACE.2019.8679362
Ramprakash, P., Sarumathi, R., Mowriya, R., & Nithyavishnupriya, S. (2020). Heart Disease Prediction Using Deep Neural Network. 2020 International Conference on Inventive Computation Technologies (ICICT), 666–670. https://doi.org/10.1109/ICICT48043.2020.9112443
Reshan, M. S. A., Amin, S., Zeb, M. A., Sulaiman, A., Alshahrani, H., & Shaikh, A. (2023). A Robust Heart Disease Prediction System Using Hybrid Deep Neural Networks. IEEE Access, 11, 121574–121591. https://doi.org/10.1109/ACCESS.2023.3328909
Shaker, C. R., Sidhartha, A., Praveena, A., Chrsity, A., & Bharati, B. (2022). An Analysis of Heart Disease Prediction using Machine Learning and Deep Learning Techniques. 2022 6th International Conference on Trends in Electronics and Informatics (ICOEI), 1484–1491. https://doi.org/10.1109/ICOEI53556.2022.9776745
Singh, A., & Jain, A. (2022). Prediction of Heart Disease using Dense Neural Network. 2022 IEEE Global Conference on Computing, Power and Communication Technologies (GlobConPT), 1–5. https://doi.org/10.1109/GlobConPT57482.2022.9938354
Sujatha, P., & Mahalakshmi, K. (2020). Performance Evaluation of Supervised Machine Learning Algorithms in Prediction of Heart Disease. 2020 IEEE International Conference for Innovation in Technology (INOCON), 1–7. https://doi.org/10.1109/INOCON50539.2020.9298354
Zhang, S., & Zhou, W. (2021). Prediction Model of Heart Failure Disease Based on GA-ELM. 2021 40th Chinese Control Conference (CCC), 7944–7948. https://doi.org/10.23919/CCC52363.2021.9549482
