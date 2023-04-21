import pandas as pd
import numpy as np
from ml import *

from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def process():
    # name = request.form.get('field')
    input_list = []
    # Retrieve data from all fields
    input_list.append(request.form.get('field1'))
    input_list.append(request.form.get('field2'))
    input_list.append(request.form.get('field3'))
    input_list.append(request.form.get('field4'))
    input_list.append(request.form.get('field5'))
    input_list.append(request.form.get('field6'))
    input_list.append(request.form.get('field7'))
    input_list.append(request.form.get('field8'))
    input_list.append(request.form.get('field9'))
    input_list.append(request.form.get('field10'))
    test_data = np.array(input_list)

    # split model
    X_train, X_test, y_train, y_test = ml_pp()

    #train cat boost model
    catboost_model = catboost(X_train, X_test, y_train, y_test)
    # Calculating the accuracies
    print("Training accuracy :", catboost_model.score(X_train, y_train))
    print("Testing accuracy :", catboost_model.score(X_test, y_test))
    # predicting the test set data
    y_pred = catboost_model.predict(test_data)
    print(test_data)
    print(input_list)
    #Clearing list
    np.delete(test_data, 0, 0)
    input_list.clear()
    print("Result : ", y_pred)
    if y_pred == 0:
        print("NEGATIVE")
        return render_template('negative.html')
    elif y_pred == 1:
        print("POSITIVE")
        return render_template('positive.html')

if __name__ == '__main__':
    app.debug = True
    app.run(port=8888)
