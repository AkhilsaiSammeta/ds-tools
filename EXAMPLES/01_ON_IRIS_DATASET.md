Here's a step-by-step practical example of a basic data science workflow using Python. We'll perform the following steps:

1. **Load a dataset**
2. **Explore the data**
3. **Preprocess the data**
4. **Train a machine learning model**
5. **Evaluate the model**
6. **Make predictions**

Let's use the famous Iris dataset for this example.

### Step 1: Load a Dataset

First, we'll load the Iris dataset from the `sklearn` library.

```python
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Convert to a pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
print(df.head())
```

### Step 2: Explore the Data

Next, we'll explore the dataset to understand its structure and basic statistics.

```python
# Basic statistics of the dataset
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize the distribution of the target variable
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='target', data=df)
plt.show()
```

### Step 3: Preprocess the Data

We'll preprocess the data by splitting it into training and test sets.

```python
from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
```

### Step 4: Train a Machine Learning Model

We'll train a logistic regression model on the training set.

```python
from sklearn.linear_model import LogisticRegression

# Initialize and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
```

### Step 5: Evaluate the Model

We'll evaluate the model's performance on the test set.

```python
from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Display a classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### Step 6: Make Predictions

Finally, we'll make predictions using the trained model on new data.

```python
# Example of new data (first three instances from the test set)
new_data = X_test.head(3)

# Make predictions
predictions = model.predict(new_data)

# Display predictions with corresponding true labels
for i, pred in enumerate(predictions):
    print(f"Prediction: {iris.target_names[pred]}, True Label: {iris.target_names[y_test.iloc[i]]}")
```

### Full Code

Here is the full code from loading the dataset to making predictions:

```python
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load a dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Step 2: Explore the data
print(df.head())
print(df.describe())
print(df.isnull().sum())
sns.countplot(x='target', data=df)
plt.show()

# Step 3: Preprocess the data
X = df.drop(columns='target')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a machine learning model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 6: Make predictions
new_data = X_test.head(3)
predictions = model.predict(new_data)
for i, pred in enumerate(predictions):
    print(f"Prediction: {iris.target_names[pred]}, True Label: {iris.target_names[y_test.iloc[i]]}")
```

This code provides a basic and practical example of a data science workflow using the Iris dataset. You can adapt and expand this workflow for more complex datasets and models.
