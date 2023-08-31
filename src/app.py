from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import pandas as pd
import joblib
import os
app = Flask(__name__)

from my_notebook_converted import search_old_cust_id
model = joblib.load(r'C:\Users\torjm\OneDrive\Bureau\project\reactproject\models\model.pkl')
QuantileTransformer = joblib.load(r'C:\Users\torjm\OneDrive\Bureau\project\reactproject\models\QuantileTransformer.pkl')
pca_Transform = joblib.load(r'C:\Users\torjm\OneDrive\Bureau\project\reactproject\models\pca.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        cust_id = request.json['cust_id']
        print(cust_id)
        cluster =search_old_cust_id(cust_id)
        return jsonify({'cluster': cluster})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/save-csv', methods=['POST'])
def save_csv():
    data = request.json  # Expecting a a single dictionary

    if not data:
        return jsonify({'error': 'No data provided'})
    # Write the single dictionary to CSV
    with open('data.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
    df = pd.read_csv('data.csv')
    a=df[['CUST_ID','PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY']]
    df=df.drop(columns=['CUST_ID','PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY'])
    df=df[['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'ONEOFF_PURCHASES_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'BALANCE_FREQUENCY', 'TENURE']]
    columns_name=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'ONEOFF_PURCHASES_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'BALANCE_FREQUENCY', 'TENURE']
    qt= pd.DataFrame(QuantileTransformer.transform(df),columns= columns_name)
    dt = pd.concat([a,qt],axis=1)
    dt=dt[['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES','ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE','PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT','TENURE']]
    a=dt['CUST_ID']
    dt=dt.drop(columns=['CUST_ID'])
    X_pca=pca_Transform.transform(dt)
    cluster=model.predict(X_pca)
    return jsonify({'cluster': cluster.tolist() , 'cust_id':a.tolist()})


CORS(app)
if __name__ == '__main__':
    app.run(port=5000)