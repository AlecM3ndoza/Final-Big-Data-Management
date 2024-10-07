# Final-Big-Data
Final 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# Load the dataset from the uploaded Excel file
file_path = '/mnt/data/Final Big Data.xlsx'
data = pd.read_excel(file_path)

# Checking for missing values in the dataset
missing_data = data.isnull().sum()
print(missing_data)

# Data summary
print(data.info())

# Distribution of income classes
sns.countplot(data['income'])
plt.title('Distribution of Income')
plt.tight_layout()
plt.show()

# Pairplot of some features
sns.pairplot(data[['age', 'hours-per-week', 'education-num', 'income']], hue='income')
plt.tight_layout()
plt.show()

# Distribution of age vs income
sns.boxplot(x='income', y='age', data=data)
plt.title('Income by Age')
plt.tight_layout()
plt.show()

# Handle missing values (simple method: drop rows with missing values)
data.dropna(inplace=True)

# Encode categorical features and the target variable
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country']
data = pd.get_dummies(data, columns=categorical_columns)
label_encoder = LabelEncoder()
data['income'] = label_encoder.fit_transform(data['income'])

# Split data into features and target
X = data.drop('income', axis=1)
y = data['income']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = logreg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Dimensionality reduction using PCA for clustering visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_pca)

# Plot the clusters
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('KMeans Clustering')
plt.tight_layout()
plt.show()

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear']}
grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluate tuned model
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))

# Feature Importance from Logistic Regression
coefficients = pd.Series(best_model.coef_[0], index=X.columns)
coefficients.sort_values().plot(kind='barh', figsize=(10, 8))
plt.title('Feature Importance from Logistic Regression')
plt.tight_layout()
plt.show()
