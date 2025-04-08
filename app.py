from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import pickle
from joblib import load
# load the model from disk
model_name = 'models/disease_detction.pkl'
clf = pickle.load(open(model_name, 'rb'))
cv=pickle.load(open('models/disease_detction.pkl','rb'))
le=pickle.load(open('models/disease_detction.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home2.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		sentence_transformed = cv.transform(data).toarray()
		prediction = clf.predict(sentence_transformed)
		predicted_label = le.inverse_transform(prediction)[0]
	return render_template('result.html',prediction = predicted_label)



if __name__ == '__main__':
	app.run(host='0.0.0.0',debug=True)