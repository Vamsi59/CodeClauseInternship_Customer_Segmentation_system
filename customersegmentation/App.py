import io
import base64
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)
data = pd.read_csv('Mall_Customers.csv')
x = data[['Annual Income (k$)', 'Spending Score (1-100)']]
k_means = KMeans(n_clusters=5, random_state=42)
    
y_means = k_means.fit_predict(x)
# Replace this with your actual machine learning code
def run_ml_code(input_data):
    result = f"Predicted clusters for user input: {y_means}"
    
    # Save the output to a file with UTF-8 encoding
    with open('ml_output.txt', 'w', encoding='utf-8') as file:
        file.write(result)

    return result, y_means

@app.route('/', methods=['GET', 'POST'])
def home():
    ml_output = scatter_plot = None
    x = None
    if request.method == 'POST':
        input_data1 = float(request.form['input1'])
        input_data2 = float(request.form['input2'])
        model=joblib.load("cust_seg")
        # Predict the cluster for the given input using the model
        prediction = model.predict([[input_data1, input_data2]])
        ml_output = f"Predicted cluster for input: {prediction[0]}"
        if prediction[0] == 0:
            x = "Customer with medium annual income and medium Annual Spend"
        elif prediction[0] == 1:
            x = "Customer with high annual income but low Annual Spend"
        elif prediction[0] == 2:
            x = "Customer with low annual income and low Annual Spend" 
        elif prediction[0] == 3:
            x = "Customer with low annual income but high Annual Spend"
        elif prediction[0] == 4:
            x = "Customer with high annual income and high Annual Spend"  
        

    return render_template('home.html', ml_output=ml_output, x=x, scatter_plot=scatter_plot)


if __name__ == '__main__':
    app.run(debug=True)
