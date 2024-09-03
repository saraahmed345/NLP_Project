Sentiment Analysis with Classical Machine Learning Models
This project implements a sentiment analysis system using various classical machine learning models. The goal is to classify movie reviews as either positive or negative. The project includes data preprocessing, model training, evaluation, and saving the best model.

Project Structure
main.py: The main script containing the code for preprocessing text, training models, and evaluating performance.
best_svm_model.pkl: The saved SVM model after training and evaluation.
README.md: Documentation for understanding and running the project.
Requirements
The project requires the following Python libraries:

bash
Copy code
numpy
nltk
scikit-learn
joblib
matplotlib
contractions
You can install the required packages using pip:

bash
Copy code
pip install numpy nltk scikit-learn joblib matplotlib contractions
Dataset
The dataset consists of movie reviews categorized into positive and negative sentiment:

Positive Reviews: Located in the txt_sentoken/pos directory.
Negative Reviews: Located in the txt_sentoken/neg directory.
Preprocessing
The preprocessing steps involve:

Tokenization: Splitting text into individual words.
Lowercasing: Converting all words to lowercase.
Punctuation Removal: Removing punctuation marks from the text.
Stopwords Removal: Filtering out common English words that don't contribute to the sentiment.
Negation Handling: Identifying and marking negated words in the text.
POS Tagging and Lemmatization: Reducing words to their base form considering their part of speech.
Contraction Expansion: Expanding contractions selectively, excluding negations.
Models Used
The following machine learning models were trained and evaluated using TF-IDF features:

Multinomial Naive Bayes
Logistic Regression
Random Forest
Support Vector Machine (SVM)
Model Training and Evaluation
TF-IDF Vectorization: Converts text into numerical features based on term frequency-inverse document frequency.
Grid Search with Cross-Validation: Fine-tunes hyperparameters for each model using 5-fold cross-validation.
Model Evaluation: Assesses model performance using accuracy, classification report, and confusion matrix.
Visualization
Model Accuracy Plot: Displays the accuracy of each model using a bar chart with a color gradient.
Best Model
The best performing model (SVM in this case) is saved as best_svm_model.pkl using joblib.

How to Run the Project
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
Run the Main Script:

Execute the script to preprocess data, train models, evaluate performance, and save the best model:

bash
Copy code
python main.py
View the Results:

After running the script, you will see the accuracy, classification report, and confusion matrix for each model, along with a plot of model accuracies.

Use the Saved Model:

You can load the saved SVM model (best_svm_model.pkl) for future predictions:

python
Copy code
import joblib
model = joblib.load('best_svm_model.pkl')
Conclusion
This project demonstrates the application of classical machine learning models for sentiment analysis. The SVM model, in particular, showed the best performance in classifying movie reviews as positive or negative.

License
This project is open-source and available under the MIT License.
