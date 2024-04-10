import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVC

# Loading the dataset

file_path = 'heart.csv'
heart_data = pd.read_csv(file_path)

# LINEAR REGRESSION
# Linear Regression for predicting cholesterol level based on age

# Feature selection for linear regression
selected_features = ['age', 'chol']
X_chol = heart_data[selected_features]
Y_chol = heart_data['chol']

# Splitting the data into training and testing sets
X_train_chol, X_test_chol, Y_train_chol, Y_test_chol = train_test_split(X_chol, Y_chol, test_size=0.2, random_state=2)

# Creating and training the linear regression model
linear_model_chol = LinearRegression()
linear_model_chol.fit(X_train_chol, Y_train_chol)

# Predicting cholesterol levels
new_data_chol = np.array([[58, 248]])  # Replace with actual values
linear_prediction_chol = linear_model_chol.predict(new_data_chol)

# Scatter plot and regression line for Age vs Cholesterol using linear regression
plt.figure(figsize=(12, 6))
sns.regplot(x=X_test_chol['age'], y=Y_test_chol, scatter_kws={'color':'red', 'label':'Cholesterol Level (Actual)'}, line_kws={'color':'blue'},ci=None, marker=None)
plt.title('Age vs Cholesterol Level (Linear Regression)')
plt.xlabel('Age')
plt.ylabel('Cholesterol Level')
plt.legend()
plt.show()

print("Linear Regression for Cholesterol Prediction")
print(f"Predicted Cholesterol Level: {linear_prediction_chol[0]:.2f} mg/dL")

# For Accuracy
# Predicting cholesterol levels on the test set
y_pred_chol = linear_model_chol.predict(X_test_chol)
# Calculate R-squared score
r2 = r2_score(Y_test_chol, y_pred_chol)
print(f"R-squared Score: {r2:.4f}") # this is accuracy denoted as R-2 Score


# LOGISTIC REGRESSION

# Feature selection for logistic regression
selected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = heart_data[selected_features]
Y = heart_data['target']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Creating and training the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)

# Predicting with more features
new_data = np.array([[65, 0, 2, 160, 360, 0, 0, 151, 0, 0.8, 2, 0, 2]])  # Replace with actual values
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

# Scatter plot and regression line with Z-Score of Age vs Heart Disease
plt.figure(figsize=(12, 6))
plt.scatter(zscore(X_test['age']), Y_test, c='red', label='Heart Problem')
sns.regplot(x=zscore(X_test['age']), y=model.predict(X_test), data=heart_data, logistic=True, ci=None, scatter=False, color='blue', line_kws={'color': 'blue'})
plt.title('Age vs Heart Disease with Z-Score (Logistic Regression)')
plt.xlabel('Age Z-score')
plt.ylabel('Heart Disease')
plt.legend()
plt.show()


print("Logistic Regression")
if prediction[0] == 0:
    print("The person does not have any heart disease.")
else:
    print("The person has heart disease.")

# Calculate accuracy score for Logistic Regression
Y_test_pred = model.predict(X_test)
accuracy_logistic_regression = accuracy_score(Y_test, Y_test_pred)
print(f"Accuracy Score of Logistic Regression: {accuracy_logistic_regression}")


# Support Vector Classifier Model (SVC)

# Feature selection
selected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = heart_data[selected_features]
Y = heart_data['target']

# Splitting the data into training and testing setsS
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and training the Support Vector Classifier (SVC) model
svc_model = SVC(kernel='linear', C=1.0)
svc_model.fit(X_train_scaled, Y_train)

# Predicting with more features
new_data = np.array([[65, 0, 2, 160, 360, 0, 0, 151, 0, 0.8, 2, 0, 2]])  # Replace with actual values
new_data_scaled = scaler.transform(new_data)
prediction = svc_model.predict(new_data_scaled)

# Scatter plot and regression line with Z-Score of Age vs Heart Disease
plt.figure(figsize=(12, 6))
plt.scatter(X_test['age'], Y_test, c='red', label='Heart Problem')
sns.regplot(x=X_test['age'], y=svc_model.predict(X_test_scaled), data=heart_data, ci=None, scatter=False, color='blue', line_kws={'color': 'blue'})
plt.title('Age vs Heart Disease with SVC')
plt.xlabel('Age')
plt.ylabel('Heart Disease')
plt.legend()
plt.show()

# Scatter plot and regression line for Age vs Heart Disease
plt.figure(figsize=(12, 6))
plt.scatter(X_test['age'], Y_test, c='red', label='Heart Problem')
sns.regplot(x=X_test['age'], y=svc_model.predict(X_test), data=heart_data, logistic=True, ci=None, scatter=False, color='blue', line_kws={'color': 'blue'})
plt.title('Age vs Heart Disease with SVC')
plt.xlabel('Age')
plt.ylabel('Heart Disease')
plt.legend()
plt.show()

# Scatter plot and regression line for Chest-Pain vs Heart Disease
plt.figure(figsize=(12, 6))
plt.scatter(X_test['cp'], Y_test, c='red', label='Heart Problem')
sns.regplot(x=X_test['cp'], y=svc_model.predict(X_test), data=heart_data, logistic=True, ci=None, scatter=False, color='blue', line_kws={'color': 'blue'})
plt.title('Chest-Pain vs Heart Disease with SVC')
plt.xlabel('Chest-Pain')
plt.ylabel('Heart Disease')
plt.legend()
plt.show()

# Scatter plot and regression line for Sex vs Heart Disease
plt.figure(figsize=(12, 6))
plt.scatter(X_test['sex'], Y_test, c='red', label='Heart Problem')
sns.regplot(x=X_test['sex'], y=svc_model.predict(X_test), data=heart_data, logistic=True, ci=None, scatter=False, color='blue', line_kws={'color': 'blue'})
plt.title('Sex vs Heart Disease with SVC')
plt.xlabel('Sex')
plt.ylabel('Heart Disease')
plt.legend()
plt.show()

print("Support Vector Classifier Model (SVC)")
if prediction[0] == 0:
    print("The person does not have any heart disease.")
else:
    print("The person has heart disease.")


# Calculate accuracy score for SVC Model
Y_test_pred_svc = svc_model.predict(X_test_scaled)
accuracy_svc = accuracy_score(Y_test, Y_test_pred_svc)
print(f"SVC Accuracy Score: {accuracy_svc}")

# Plotting Bar Graph to compare Models
models = ['Logistic Regression', 'SVC']

# Accuracy scores
accuracy_scores = [accuracy_logistic_regression, accuracy_svc]

# Plotting bar graph
plt.figure(figsize=(10, 6))
plt.bar(models, accuracy_scores, color=['orange', 'green'])
plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1 for accuracy scores
plt.title('Comparison of Accuracy Scores')
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.show()
