import os
import nltk
import numpy as np
import string
import contractions
import warnings
import joblib 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Define negation handling function
def handle_negation(text):
    negations = {"not", "no", "never", "n't"}
    negated = False
    words = []
    for word in text.split():
         if word in negations:
            negated = not negated
         else:
            if not negated:
                words.append(word)
            else:
                words.append("-" + word)  # Append with negation symbol
         negated = False  # Reset negation for the next word
    return " ".join(words)


# Define preprocessing function
def preprocess_text(text):
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())

    # Remove punctuation and handle numbers
    tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in tokens if not word.isdigit()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Negation handling
    tokens = handle_negation(text=" ".join(tokens)).split()
    # POS tagging and lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token, pos in pos_tag(tokens):
        if pos.startswith('N'):  # Nouns
            lemma = lemmatizer.lemmatize(token, pos='n')
        elif pos.startswith('V'):  # Verbs
            lemma = lemmatizer.lemmatize(token, pos='v')
        elif pos.startswith('J'):  # Adjectives
            lemma = lemmatizer.lemmatize(token, pos='a')
        else:
            lemma = token
        lemmatized_tokens.append(lemma)

    # Expand contractions selectively (same as before)
    expanded_tokens = [contractions.fix(token) if token.lower() not in ['not', 'no'] else token for token in
                       lemmatized_tokens]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(expanded_tokens).strip()

    return preprocessed_text


# Ignore warnings
warnings.filterwarnings("ignore")

# Directory containing review files
positive_reviews_dir = r'C:\Users\DELL\OneDrive\Desktop\New folder\txt_sentoken\pos'
negative_reviews_dir = r'C:\Users\DELL\OneDrive\Desktop\New folder\txt_sentoken\neg'

# Preprocess reviews
preprocessed_positive_reviews = []
preprocessed_negative_reviews = []

# Iterate over each file in the negative reviews directory
for filename in os.listdir(negative_reviews_dir):
    with open(os.path.join(negative_reviews_dir, filename), 'r', encoding='utf-8') as file:
        review = file.read()
        preprocessed_review = preprocess_text(review)
        preprocessed_negative_reviews.append(preprocessed_review)

# Iterate over each file in the positive reviews directory
for filename in os.listdir(positive_reviews_dir):
    with open(os.path.join(positive_reviews_dir, filename), 'r', encoding='utf-8') as file:
        review = file.read()
        preprocessed_review = preprocess_text(review)
        preprocessed_positive_reviews.append(preprocessed_review)

# Combine positive and negative preprocessed reviews
all_reviews = preprocessed_positive_reviews + preprocessed_negative_reviews
labels = ['positive'] * len(preprocessed_positive_reviews) + ['negative'] * len(preprocessed_negative_reviews)

print("*************")
print("preprocessed_positive_reviews: ")
print(preprocessed_positive_reviews[0])
print("******************")
print("******************")
print("preprocessed_negative_reviews: ")
print(preprocessed_negative_reviews[0])
print("*************")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_reviews, labels, test_size=0.2, random_state=150)

# Create pipelines with TF-IDF vectorizer and different classifiers
pipelines = {
    'Multinomial Naive Bayes': make_pipeline(TfidfVectorizer(), MultinomialNB()),
    'Logistic Regression': make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)),
    'Random Forest': make_pipeline(TfidfVectorizer(), RandomForestClassifier()),
    'SVM': make_pipeline(TfidfVectorizer(), SVC()),
}

# Define parameter grids for grid search
param_grids = {
    
    'Multinomial Naive Bayes': {},
    'Logistic Regression': {'logisticregression__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    'Random Forest': {'randomforestclassifier__n_estimators': [100, 200, 300],
                      'randomforestclassifier__max_depth': [None, 10, 20]},
    'SVM': {'svc__C': [0.1, 1.0, 10.0], 'svc__kernel': ['linear', 'rbf'],
            'svc__gamma': ['scale', 'auto']},
}


# Perform grid search and fit models
model_accuracies = {}

for name, pipeline in pipelines.items():
    grid_search = GridSearchCV(pipeline, param_grid=param_grids[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

     # Calculate accuracy for the entire test set
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 2)

    # Store model accuracy
    model_accuracies[name] = accuracy
    
    print("###################################################################")
    print(f"Model: {name}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print("###################################################################")



# Calculate color gradient based on accuracies
colors = cm.viridis(np.linspace(0, 1, len(model_accuracies)))

# Visualize accuracies of models with color gradient
plt.figure(figsize=(10, 6))
plt.bar(model_accuracies.keys(), model_accuracies.values(), color=colors)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracies')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Save the SVM model
svm_model = pipelines['SVM']  
svm_grid_search = GridSearchCV(svm_model, param_grid=param_grids['SVM'], cv=5, scoring='accuracy')  # Re-fit with best parameters
svm_grid_search.fit(X_test, y_test)
best_svm_model = svm_grid_search.best_estimator_
joblib.dump(best_svm_model, 'best_svm_model.pkl')  
print("SVM model saved successfully as 'best_svm_model.pkl'.")