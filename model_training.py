from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

def train_model(corpus, labels, model_type='svm', ngram_range=(1, 2), max_features=200, test_size=0.3, random_state=0):
    """
    Trains a machine learning model using TF-IDF vectorization on the provided corpus and labels.

    Parameters:
    corpus (list): A list of preprocessed text data.
    labels (numpy.ndarray): An array of target labels.
    model_type (str, optional): The type of model to train. Options: 'svm', 'random_forest', 'naive_bayes'.
                                Default is 'svm'.
    ngram_range (tuple, optional): The n-gram range for TF-IDF vectorization. Default is (1, 2).
    max_features (int, optional): The maximum number of features for TF-IDF vectorization. Default is 200.
    test_size (float, optional): The proportion of the test set. Default is 0.3.
    random_state (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
    object: The trained model.
    """
    cv = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    x = cv.fit_transform(corpus).toarray()

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)

    if model_type == 'svm':
        model = SVC()
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
    else:
        raise ValueError("Invalid model_type. Available options: 'svm', 'random_forest', 'naive_bayes'")

    model.fit(x_train, y_train)

    return model