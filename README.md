import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load the dataset
df = pd.read_csv("fake_or_real_news.csv")  # Replace with your actual dataset path
print("Dataset Loaded. Sample:")
print(df.head())

# 2. Basic preprocessing
df = df[['text', 'label']]  # Assuming 'text' and 'label' columns exist
df.dropna(inplace=True)

# 3. Encode labels
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 5. Convert text to numeric using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 7. Predictions and Evaluation
y_pred = model.predict(X_test_vec)

print("\n--- Evaluation Metrics ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Predict on new input
def predict_news(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    return "REAL" if prediction[0] == 1 else "FAKE"

# Example
news_input = input("\nEnter a news article to check if it's real or fake:\n")
print("Prediction:", predict_news(news_input))
