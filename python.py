python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
import os

nltk.download('punkt')
nltk.download('stopwords')

# Function to process the email content
def process_email(email_content):
    tokens = word_tokenize(email_content.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words

# Function to create a frequency dictionary for the words in emails
def create_word_frequency(emails, is_spam):
    word_frequency = defaultdict(int)
    for email in emails:
        words = process_email(email)
        for word in words:
            word_frequency[word] += 1 if is_spam else -1
    return word_frequency

# Example emails (You'd typically load your email dataset here)
spam_emails = [
    "Get a free iPad now!",
    "Make money fast with this amazing opportunity!",
    # Add more spam emails here
]

ham_emails = [
    "Meeting agenda for next week",
    "Reminder: Your appointment tomorrow",
    # Add more legitimate emails here
]

# Create word frequency dictionaries for spam and ham emails
spam_word_frequency = create_word_frequency(spam_emails, True)
ham_word_frequency = create_word_frequency(ham_emails, False)

# Example email to test
test_email = "Congratulations! You've won a lottery."

# Process the test email
test_words = process_email(test_email)

# Calculate probability of the test email being spam
spam_probability = 0
for word in test_words:
    spam_probability += spam_word_frequency[word]

if spam_probability > 0:
    print("This email is likely spam.")
else:
    print("This email seems legitimate.")
