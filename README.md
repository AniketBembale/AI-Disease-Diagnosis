# AI-Disease-Diagnosis

### Objective
The main objective of this project is to develop a machine learning model using NLP techniques for disease detection based on symptoms and achieve high accuracy in classifying diseases.

### Description
This project involves the development of a machine learning model for disease detection based on symptoms using natural language processing (NLP) techniques. The process includes data pre-processing, exploratory data analysis, feature engineering, model training using Support Vector Machine (SVM) and neural network algorithms, hyperparameter tuning, and deployment using the Flask framework.

The model achieved impressive results, with a training accuracy of 99% and testing accuracy of 93%. To preprocess the textual data, various NLP techniques were employed, such as lemmatization, stop-word removal, and TF-IDF vectorization. Class labels were encoded using a label encoder. The model was deployed using Flask, allowing for easy interaction with end-users.

### Technology
The project was developed using the following technologies:
- Python
- Natural Language Processing (NLP)
- Scikit-learn
- TensorFlow
- Flask
- HTML

### Repository Structure
```
|-- README.md
|-- data_preprocessing.py
|-- model_training.py
|-- model_evaluation.py
|-- app.py
|-- requirements.txt
|-- data/
|   |-- Symptom2Disease.csv
|-- models/
|   |-- disease_detection_model.pkl
|   |-- transform.pkl
|   |-- lebal_encoder.pkl
|-- templates/
|   |-- index.html
|   |-- home.html
```

### Instructions
1. Clone the repository: `git clone https://github.com/yourusername/disease-detection.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run data preprocessing: `python data_preprocessing.py`
4. Train the model: `python model_training.py`
5. Evaluate the model: `python model_evaluation.py`
6. Start the Flask web application: `python app.py`
7. Access the web application at `http://localhost:5000` in your browser.

### Notes
- Ensure you have Python 3.6+ installed.
- The raw data is stored in the 'data/raw_data.csv' file, and the preprocessed data is stored in 'data/preprocessed_data.csv'.
- The trained model is saved in 'models/disease_detection_model.pkl'.

Feel free to explore the code and customize it to fit your specific use case. If you have any questions or feedback, please don't hesitate to contact us.

**Contributors:**  
Aniket Bembale (anibembale1004@gmail.com)  
