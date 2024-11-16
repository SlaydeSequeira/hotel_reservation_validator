import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('Hotel Reservations.csv')

# Prepare the data
# Drop irrelevant columns if needed (none specified here), and identify the target and feature columns
X = df.drop(columns=['booking_status'])  # Features
y = df['booking_status'].apply(lambda x: 1 if x == 'Confirmed' else 0)  # Target: encode as 1 for Confirmed, 0 otherwise

# Check for class balance
print("Class distribution in target:", y.value_counts())

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Apply Label Encoding to each categorical column
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Store each encoder if needed later

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and label encoders to a .pkl file
with open('/mnt/data/booking_status_model.pkl', 'wb') as file:
    pickle.dump((model, label_encoders), file)
