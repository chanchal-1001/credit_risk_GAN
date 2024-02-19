#import sys
#sys.path.append('/project/app/main/project/utils')
import sys
sys.path.append("..")

import time
from utils import credit_utils
import binary_classifier
#from binary_classifier import BinaryClassifier
import credit_risk_model


def run():
    start_time = time.time()
    #Load Credit Risk data in batched form
    encrypted_df_hot, encrypted_batches = credit_risk_model.loadEncryptedBatchesForGan()
    #Splitting enctrypted_df_hot into training and test
    X_train, X_test, y_train, y_test = credit_utils.split_encrypted_data(encrypted_df_hot)
    print("######################### Building GAN Model for German Credit Data  ###############################")
    gan, generator =  credit_risk_model.trainGanModel(encrypted_batches)
    #Save Gan Model
    file_path = credit_risk_model.saveGanModel(gan)
    #load the gan model
    credit_risk_model.loadGanModel(file_path)
        
    #Generate Synthetic data using trained Generator
    features_and_target = encrypted_batches[0].columns
    binary_gan_values_df = credit_risk_model.generateSyntheticDataUsingGan(generator, features_and_target)    
    #Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")
    
    print("########################## Genrating and Training Binary Classifer ###########################################")
    #Removing Target feature from columns
    features = features_and_target[:-1]
    #Creating Binary 
    binary_classifier_model, criterion, optimizer = binary_classifier.createBinaryClassifierModel(features)
    #Training the classifier on X_train and y_train encrypted features
    trained_binary_classifier_model = binary_classifier.trainingTheClassifier(binary_classifier_model, criterion, optimizer,X_train, y_train)
    #Saving the Model
    binary_classifier.saveBinaryClassifierModel(trained_binary_classifier_model)
    #loading the saved model
    binary_credit_risk_model = binary_classifier.loadBinaryClassifierModel()
    #Testing the binary classifiers
    binary_classifier.testBinaryClassifierModel(binary_credit_risk_model, X_test, y_test)


if __name__ == '__main__':
    run()