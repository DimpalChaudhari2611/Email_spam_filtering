import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


emails = [
    ("Get a free vacation now!", "spam"),
    ("Meeting at 10 am tomorrow.", "ham"),
    # Add more emails with their classifications
]

# Tokenization, stop words removal, and lemmatization
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(filtered_tokens)

processed_emails = [(preprocess_text(text), label) for text, label in emails]

# Extract features using TF-IDF
texts = [text for text, label in processed_emails]
labels = [label for text, label in processed_emails]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predictions and evaluation
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
