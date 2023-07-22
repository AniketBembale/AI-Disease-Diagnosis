from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

def preprocess_text(data_frame):
    """
    Preprocesses the text in the given DataFrame by removing non-alphabetic characters, converting to lowercase,
    lemmatizing words, and removing stop words.

    Parameters:
    data_frame (pandas.DataFrame): The DataFrame containing the 'text' column to be preprocessed.

    Returns:
    list: A list containing the preprocessed text data.
    """
    lem = WordNetLemmatizer()
    corpus = []
    for i in range(len(data_frame)):
        review = re.sub("[^a-zA-Z]", " ", data_frame["text"][i])
        review = review.lower()
        review = review.split()
        review = [lem.lemmatize(word) for word in review if word not in set(stopwords.words("english"))]
        review = " ".join(review)
        corpus.append(review)
    return corpus
