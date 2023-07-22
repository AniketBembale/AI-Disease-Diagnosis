from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, x_test, y_test):
    """
    Evaluates the performance of a trained machine learning model using test data.

    Parameters:
    model (object): The trained machine learning model.
    x_test (numpy.ndarray): Test features.
    y_test (numpy.ndarray): Test labels.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    evaluation_metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }

    return evaluation_metrics