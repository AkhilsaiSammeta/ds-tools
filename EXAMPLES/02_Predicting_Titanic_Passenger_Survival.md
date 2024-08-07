# Predicting Titanic Passenger Survival: A Step-by-Step Data Science Workflow with Python
Here's step-by-step practical data science example using Python. This time, we'll use the Titanic dataset from Kaggle to predict the survival of passengers. The steps will be similar but with a different dataset and additional preprocessing steps.

### Step 1: Load the Dataset

First, we'll load the Titanic dataset.

```python
import pandas as pd

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

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

sns.countplot(x='Survived', data=df)
plt.show()
```

### Step 3: Preprocess the Data

We'll preprocess the data by handling missing values, encoding categorical variables, and splitting it into training and test sets.

```python
# Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop columns that won't be used in the analysis
df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Split the data into training and test sets
from sklearn.model_selection import train_test_split

X = df.drop(columns='Survived')
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
```

### Step 4: Train a Machine Learning Model

We'll train a Random Forest classifier on the training set.

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
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
print(classification_report(y_test, y_pred))
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
    print(f"Prediction: {pred}, True Label: {y_test.iloc[i]}")
```

### Full Code

Here is the full code from loading the dataset to making predictions:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Step 2: Explore the data
print(df.head())
print(df.describe())
print(df.isnull().sum())
sns.countplot(x='Survived', data=df)
plt.show()

# Step 3: Preprocess the data
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df.drop(columns='Survived')
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# Step 6: Make predictions
new_data = X_test.head(3)
predictions = model.predict(new_data)
for i, pred in enumerate(predictions):
    print(f"Prediction: {pred}, True Label: {y_test.iloc[i]}")
```

This example demonstrates a practical workflow for predicting passenger survival on the Titanic using a Random Forest classifier. You can further refine the preprocessing steps, model selection, and evaluation techniques based on your specific requirements.
