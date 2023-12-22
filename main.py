import pandas as pd
import numpy as np
import nltk
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

print("Initializing Application. Please wait...")

# Download necessary nltk stopwords extension
nltk.download('stopwords')
# Make the import from here, os this does not fail if it was not installed before.
from nltk.corpus import stopwords
# Cache stop words, prevent loading them multiple times.
cached_stop_words = stopwords.words("english")


def clean_text(text):
    global cached_stop_words
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in cached_stop_words])
    return text

def text_to_vector(text):
    global words_size
    word_vector = np.zeros(words_size)
    for word in text.split(" "):
        if word_dict.get(word) is None:
            continue
        else:
            word_vector[word_dict.get(word)] += 1
    return np.array(word_vector)


# read csv with pandas
data  = pd.read_csv("csv_files/Equal_neutral_reviews_removed.csv", sep=",", usecols=["Summary", "Sentiment"],  encoding='unicode_escape')

# Replace positive/negative with binary identifiers.
data = data.replace(['positive','negative'],[0, 1])

# clean up summaries
data['Summary'] = data['Summary'].apply(clean_text)


# set the data frames
text = pd.DataFrame(data['Summary'])
label = pd.DataFrame(data['Sentiment'])



# Count word frequency
total_counts = Counter()
for i in range(len(text)):
    for word in text.values[i][0].split(" "):
        total_counts[word] += 1

# Sort by the most frequently used words
words_sorted = sorted(total_counts, key=total_counts.get, reverse=True)

words_size = len(words_sorted)
word_dict = {}
for i, word in enumerate(words_sorted):
    word_dict[word] = i

word_vectors = np.zeros((len(text), len(words_sorted)), dtype=np.int_)
for i, (_, raw_text) in enumerate(text.iterrows()):
    word_vectors[i] = text_to_vector(raw_text.iloc[0])

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data['Summary'])

# https://www.milindsoorya.com/blog/build-a-spam-classifier-in-python
X_train, X_test, y_train, y_test = train_test_split(vectors, data['Sentiment'], test_size=0.15, random_state=111)

#initialize mnb classification model
mnb = MultinomialNB(alpha=0.2)
# Begin the training
mnb.fit(X_train, y_train)
pred = mnb.predict(X_test)

# https://ashejim.github.io/C964/task2_c/example_sup_class/sup_class_ex-accuracy.html
# Output accuracy score.
print("Model Accuracy: " + str(accuracy_score(y_test , pred)))

# https://ashejim.github.io/C964/task2_c/example_sup_class/sup_class_ex-accuracy.html
# Output average score
k_folds = KFold(n_splits = 5, shuffle=True)
scores = cross_val_score(mnb, vectors, data['Sentiment'])
print("Average Score: ", scores.mean())

# https://ashejim.github.io/C964/task2_c/example_sup_class/sup_class_ex-accuracy.html
# Output logicistic regression score
log_model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
predictions_log = log_model.predict(X_test)
lr_score = accuracy_score(y_test, predictions_log)
print("Regression Score: ", lr_score)

# # # https://ashejim.github.io/C964/task2_c/example_sup_class/sup_class_ex-accuracy.html
# ConfusionMatrixDisplay.from_estimator(mnb, X_test, y_test)
# cm = confusion_matrix(y_test, pred, labels=mnb.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mnb.classes_)
# disp.plot()
# plt.show()


print()
print("---------------------------------------------")
print("Welcome to the product review analysis tool!")
print("Enter a review below and hit ENTER.")
print("Hit ENTER with a blank input to exit.")

while True:
    text_input = input("Enter product review: ")
    if text_input == "":
        print("Thank you for using the program! Now exiting...")
        quit()
    else:
        text = [text_input]
        integers = vectorizer.transform(text)
        x = mnb.predict(integers)
        
        if x == 0:
            print("Analysis Result: Positive Review")
        else:
            print("Analysis Result: Negative Review")
