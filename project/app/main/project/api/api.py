import sys
sys.path.append("..")
from flask import Flask, request, jsonify
import pandas as pd
import torch
import numpy as np
from utils import credit_utils

from services import binary_classifier


app = Flask(__name__)

@app.route('/predictCreditRisk', methods=['POST'])
def predict():
    try:
        print('Executing predictCreditRisk API')
        # Get the input data from the request
        input_data = request.get_json(force=True)                 
        # Convert JSON data to DataFrame      
        df = pd.DataFrame(input_data)        
        input_data_hot = credit_utils.mapMyData(df, X_hot_columns)   
        #Predicting credit risk using binary classifer model
        y_pred = CR_model(torch.tensor(input_data_hot.values, dtype=torch.float)) 
        y_pred_binary = np.where(y_pred > 0.5  , 1, 2)        
        return jsonify(y_pred_binary.tolist())        
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})
    

    
if __name__ == "__main__":    
    #loading Binary Classier Model
    CR_model = binary_classifier.loadBinaryClassifierModel()
    #Initialize columns for mapping
    X_hot_columns = credit_utils.getX_hot_columns()    
    app.run(debug=False, port=8049, use_reloader=False)    