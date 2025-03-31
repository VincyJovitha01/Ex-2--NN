<H3>Experiment No. 2 Implementation of Perceptron for Binary Classification</H3>
<H3>Name : Vincy Jovitha V</H3>
<H3>Register no: 212223230242</H3>
<H3>Date: 31.03.2025</H3>

### AIM:
To implement a perceptron for classification using Python<BR>

### EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

### RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


### ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to 0 or 1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>

### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self._b = 0.0
        self._w = None
        self.misclassified_samples = []

    def fit(self, x: np.array, y: np.array, n_iter=20):
        self._b = 0.0
        self._w = np.zeros(x.shape[1])
        self.misclassified_samples = []

        for _ in range(n_iter):
            errors = 0
            for xi, yi in zip(x, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)
            self.misclassified_samples.append(errors)

    def f(self, x: np.array) -> float:
        return np.dot(x, self._w) + self._b

    def predict(self, x: np.array):
        return np.where(self.f(x) >= 0, +1, -1)  # Output is +1 or -1

# Load the dataset
df = pd.read_csv('anemia.csv')
print(df.head())

# Drop the 'Gender' column if it exists
if 'gender' in df.columns:
    df = df.drop(columns=['gender'])

# Identify features (all except the last column, which is the target)
target_col = df.columns[-1]
x = df.drop(columns=[target_col]).values
y = df[target_col].values

# Convert target values: Change 0 to -1 (ensuring only +1 and -1)
y = np.where(y == 0, -1, +1)

# Standardize the feature values
x = (x - x.mean(axis=0)) / x.std(axis=0)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Create and train the perceptron
classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train, n_iter=20)

# Evaluate the model
accuracy = accuracy_score(classifier.predict(x_test), y_test) * 100
print("Accuracy:", accuracy)

# 2D Scatter Plot (First Two Features)
plt.figure(figsize=(8, 6))
plt.scatter(x[anemic, 0], x[anemic, 1], color='red', marker='o', label='Anemic (+1)')
plt.scatter(x[not_anemic, 0], x[not_anemic, 1], color='blue', marker='x', label='Not Anemic (-1)')

plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title("2D Scatter Plot of First Two Features")
plt.legend()
plt.show()

# Plot the errors during training
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(classifier.misclassified_samples) + 1), 
         classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.title('Errors vs Epoch')
plt.show()

# 3D Scatter Plot (First Three Features)
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')

# Get indices for each class
anemic = (y == +1)
not_anemic = (y == -1)

ax.scatter(x[anemic, 0], x[anemic, 1], x[anemic, 2], 
           color='red', marker='o', label='Anemic (+1)')
ax.scatter(x[not_anemic, 0], x[not_anemic, 1], x[not_anemic, 2], 
           color='blue', marker='^', label='Not Anemic (-1)')

ax.set_title('Anemia Dataset (3D View)')
ax.set_xlabel(df.columns[0] + " (standardized)")
ax.set_ylabel(df.columns[1] + " (standardized)")
ax.set_zlabel(df.columns[2] + " (standardized)")
plt.legend()
plt.show()
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/542af695-4da2-4802-acc7-e80be3c32ac6)
![image](https://github.com/user-attachments/assets/9f5bb1a3-080d-4cf2-9002-e23789d5586f)
![image](https://github.com/user-attachments/assets/37a87468-bf8e-4c43-a73f-af903ab951a7)
![image](https://github.com/user-attachments/assets/56c0e4ee-8fbe-496e-997c-36497d8c1831)

### RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
