# Predict salary based on year

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.externals import joblib

from flask import Flask, request
from flask_restful import Resource, Api
from flask import jsonify

import json

app = Flask(__name__)
api = Api(app)

class LinearRegressionAPI(Resource):
    def post(self):
        try:
            # Get the year from the post value
            # and type cast as float
            year =  float(request.values.get('year'))

            # Importing the dataset
            dataset = pd.read_csv('Salary_Data.csv')
            X = dataset.iloc[:, :-1].values
            Y = dataset.iloc[:, 1].values

            # Splitting the dataset into the Training set and Test set
            X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size = 1/3, random_state = 0
            )

            # Fitting Simple Linear Regression to the Training set
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)

            # Predicting the Test set results
            y_pred = regressor.predict(X_test)

            # Call API
            joblib.dump(regressor, 'model.pkl')
            lr = joblib.load('model.pkl')

            # Predict salary for the posted years
            return jsonify(lr.predict([[year]]).tolist())
        except Exception as e:
            data = {}
            # Better pproach is log the error and return something went wrong.
            data['message'] = str(e)
            json_data = json.dumps(data)
            return json_data, 500

api.add_resource(LinearRegressionAPI, '/')

if __name__ == '__main__':
    app.run(debug=True)
