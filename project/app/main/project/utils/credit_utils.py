import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder
import os
from services.credit_risk_encryption import encrypt_deterministic
import services.credit_risk_encryption as CRE
from sklearn.model_selection import train_test_split


#Method to convert input_df to onehot encoded form to be consumed by model
def mapMyData(input_df, X_hot_columns):
    names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker']
    num_col = ["duration","creditamount", "installmentrate", "residencesince", "age","existingcredits","peopleliable"]
    
   
    encrypted_out_cols = X_hot_columns                         
    output_df = pd.DataFrame(columns=names)
    if (type(input_df) != pd.DataFrame):
        raise ValueError ("Hey, Cannot find dataframe object")
    elif (len(input_df.columns) <20 or len(input_df.columns) >20):
        raise ValueError ("Hey, The data entered is incorrect. Please check all columns properly")
    for col in input_df.columns:
        if col not in names:
            raise ValueError ("Hey, Unidentified Colums in the input Data. Please correct data.")
    num_map = {'duration': [ 4., 21., 38., 55., 72.],
                 'creditamount': [  250. ,  4793.5,  9337. , 13880.5, 18424. ],
                 'installmentrate': [1.  , 1.75, 2.5 , 3.25, 4.  ],
                 'residencesince': [1.  , 1.75, 2.5 , 3.25, 4.  ],
                 'age': [19., 33., 47., 61., 75.],
                 'existingcredits': [1.  , 1.75, 2.5 , 3.25, 4.  ],
                 'peopleliable': [1.  , 1.25, 1.5 , 1.75, 2.  ]}
    Label_map = {'existingchecking': {'A11': 0, 'A12': 1, 'A14': 3, 'A13': 2},
                 'duration': {},
                 'credithistory': {'A34': 4, 'A32': 2, 'A33': 3, 'A30': 0, 'A31': 1},
                 'purpose': {'A43': 4,
                  'A46': 7,
                  'A42': 3,
                  'A40': 0,
                  'A41': 1,
                  'A49': 9,
                  'A44': 5,
                  'A45': 6,
                  'A410': 2,
                  'A48': 8},
                 'creditamount': {},
                 'savings': {'A65': 4, 'A61': 0, 'A63': 2, 'A64': 3, 'A62': 1},
                 'employmentsince': {'A75': 4, 'A73': 2, 'A74': 3, 'A71': 0, 'A72': 1},
                 'installmentrate': {},
                 'statussex': {'A93': 2, 'A92': 1, 'A91': 0, 'A94': 3},
                 'otherdebtors': {'A101': 0, 'A103': 2, 'A102': 1},
                 'residencesince': {},
                 'property': {'A121': 0, 'A122': 1, 'A124': 3, 'A123': 2},
                 'age': {},
                 'otherinstallmentplans': {'A143': 2, 'A141': 0, 'A142': 1},
                 'housing': {'A152': 1, 'A153': 2, 'A151': 0},
                 'existingcredits': {},
                 'job': {'A173': 2, 'A172': 1, 'A174': 3, 'A171': 0},
                 'peopleliable': {},
                 'telephone': {'A192': 1, 'A191': 0},
                 'foreignworker': {'A201': 0, 'A202': 1},
                 'classification': {1: 0, 2: 1}}
    #Iterate over input DataFrame
    for col in input_df.columns:
        
            if col in num_col:
                output_array = []
                for value in input_df[col]:
                    val_arr = num_map[col]
                    ret_val = 0.0
                    for pos in range(len(val_arr)):
                        if (value > val_arr[pos] and pos>0):
                            ret_val = float(pos)
                    output_array.append(ret_val)
                    #print(col, value, ret_val)
                output_df[col] = output_array
                
            else:
                output_array = []
                for value in input_df[col]:
                    #print(col, value, Label_map[col][value])
                    output_array.append(Label_map[col][value])
                output_df[col] = output_array
    key  = CRE.getKey()
    encrypted_df = output_df.apply(lambda x: x.apply(lambda y: CRE.encrypt_deterministic(key, y)))
    output_df_hot = pd.get_dummies(encrypted_df, columns=encrypted_df.columns)
    encrypted_out_hot = pd.DataFrame(0,columns=encrypted_out_cols, index = range(input_df.shape[0]))
    for col in encrypted_out_hot.columns:
        if col in output_df_hot.columns:
            encrypted_out_hot[col]=output_df_hot[col]
            #print("Hey", col)
    return encrypted_out_hot
    
    
def loadCreditRiskData():
    print("Load the German credit risk data")
    folder_path = '../../resource/data/'
   
    data_file_name = 'german.data'

    # Construct the full path to the CSV file
    file = os.path.join(folder_path, data_file_name)
    
    names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
             'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
             'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
             'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

    data = pd.read_csv(file, names = names, delimiter=' ')
    
    return getEncodedData(data)

def batchEncryptedData(encrypted_df_hot, batch_size):    
    
    encrypted_batches = [encrypted_df_hot[i:i + batch_size] for i in range(0,len(encrypted_df_hot), batch_size)]    
    return encrypted_batches

def split_encrypted_data(encrypted_df_hot):   
    X_hot = encrypted_df_hot.drop('classification', axis=1)
    Y = encrypted_df_hot["classification"]
    print(X_hot.shape)
    print(Y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_hot,Y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def getEncodedData(data):    
    # Create a label encoder
    lb_encoder = LabelEncoder()
    est = KBinsDiscretizer(n_bins=4, encode='ordinal', 
                           strategy='uniform')
    num_col = ["duration","creditamount", "installmentrate", "residencesince", "age","existingcredits","peopleliable"]
    for col in num_col:    
        data[col] = est.fit_transform(data[[col]])
    
    for item in data.columns:
        if item not in num_col:            
            data[item] = lb_encoder.fit_transform(data[item]) 
    return data



def getEncrypted_df_hot(X_hot):
    encrypted_df_hot = X_hot.copy()
    encrypted_df_hot["classification"] = Y
    return encrypted_df_hot


def getX_hot_columns():
    # Open the file in read mode
    folder_path = '../../resource/data/'
    with open(folder_path + 'columns.txt', 'r') as file:
    # Read the content of the file as a string
        content = file.read()

        # Split the string into a list of strings using space as the separator
        numbers_as_strings = content.split(' ')

        # Convert the list of strings to a list of integers
        X_hot_columns = numbers_as_strings #list(map(int, numbers_as_strings))    
    return X_hot_columns
    
def get_encrypted_features(data):    

    X = get_X(data)
    Y = get_Y(data)    
    key = CRE.getKey()                        
    encrypted_df = X.apply(lambda x: x.apply(lambda y: CRE.encrypt_deterministic(key, y)))    
    X_hot = pd.get_dummies(encrypted_df, columns=encrypted_df.columns)
    encrypted_df_hot = X_hot.copy()
    encrypted_df_hot["classification"] = Y      
    X_hot_columns = X_hot.columns
    
    folder_path = '../../resource/data/'
    with open(folder_path+'columns.txt', 'w') as file:
        # Convert the elements of the array to strings and join them into a single string
        array_as_string = ' '.join(map(str, X_hot_columns))

        # Write the string to the file
        file.write(array_as_string)
    return encrypted_df_hot

def get_X(data):
    X = data[['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
             'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
             'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
             'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker']]
    return X

def get_Y(data):
    Y = data["classification"]
    return Y