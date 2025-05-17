# Batch-04
Exposing the truth with advanced fake news detection powered by natural language processing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('data/fake_or_real_news.csv')
X = df['text']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vect = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save model and vectorizer
with open('models/fake_news_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)
