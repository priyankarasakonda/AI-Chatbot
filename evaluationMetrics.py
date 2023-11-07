import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support



# Load data
with open('intents.json', 'r') as f:
    data = json.load(f)

# Prepare data
X = []  # input messages
y = []  # intent labels
for intent in data['intents']:
    for pattern in intent['patterns']:
        X.append(pattern.lower())
        y.append(intent['tag'])
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_true = y_test
# Train model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)
precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=1)
# Calculate evaluation metrics
print('Confusion matrix:')

print(classification_report(y_true, y_pred, zero_division=0))

print(confusion_matrix(y_test, y_pred))
print('Classification report:')
print(classification_report(y_test, y_pred))
