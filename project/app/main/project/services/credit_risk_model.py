import sys
sys.path.append("..")

import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout, Flatten, Activation
from keras.optimizers.legacy import Adam
from keras.layers import Input
from keras.models import Model
from utils import credit_utils
import numpy as np
import pandas as pd
batch_size = 50

#Define Generator
def create_generator(input_size):
    generator = Sequential()
    generator.add(Dense(units=128, input_dim=input_size))    
    generator.add(Dense(units=256))    
    generator.add(Dense(units=512))
    generator.add(Dense(units=1024))  
    generator.add(Dense(units=input_size, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0000001, beta_1=0.3, beta_2=0.5))
    return generator

#Define discriminator
def create_discriminator(input_size):
    discriminator = Sequential()
    discriminator.add(Dense(units=128, input_dim=input_size))
    discriminator.add(Dense(units=256))
    discriminator.add(Dense(units=512))  
    discriminator.add(Dense(units=1024, activation='relu'))
    discriminator.add(Dense(units=input_size, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0000001, beta_1=0.3, beta_2=0.5))
    return discriminator

# create GANs 
def create_gan(discriminator, generator, input_size):
    print("Defining the Architechture of GAN Model")
    discriminator.trainable = False
    gan_input = Input(shape=(input_size,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss= 'binary_crossentropy', optimizer='adam', metrics='accuracy')
    gan.summary()
    return gan

#define training the GAN
def train_gan(gan, generator, discriminator,encrypted_batches, tot_columns, epochs=1, batch_size=50):
    print("Training the GAN Model")
    for e in range(epochs):
        print("In epoch :::::::::::::::::::::::::::::::::::::::::::::::::: ", e)        
        for batch in encrypted_batches:            
            # print("In batch no :::::::::::::::::::::::::::::::::::::::::::::::::: ", i)
            noise = np.random.normal(0,1,[batch_size,tot_columns])
            generated_data = generator.predict(noise)
            
            real_data = batch
            #real_data = np.stack(real_data, axis=0)
            discriminator.trainable = True
            #Compute the discriminator's loss on real data
            real_loss= discriminator.train_on_batch(real_data,np.ones((batch_size,tot_columns)))
            #Compute the discriminator's loss on fake data           
            fake_loss= discriminator.train_on_batch(generated_data,np.zeros((batch_size,tot_columns))) 
            discriminator.trainable = False #Don't change discriminator weights
            loss, accuracy = gan.train_on_batch(noise, np.ones((batch_size,tot_columns)))
            #loss, accuracy = gan.train_on_batch(X_test, y_test)
            print('loss: ', loss)            
            if loss < .4:
                break
        if loss < .4:
            break    
       
    print("Completed Training the GAN Model")
      

def loadEncryptedBatchesForGan():
    data = credit_utils.loadCreditRiskData()
    # Record the start time
    #start_time = time.time()
    encrypted_df_hot = credit_utils.get_encrypted_features(data)
  
    encrypted_batches  = credit_utils.batchEncryptedData(encrypted_df_hot, 50)
    print(len(encrypted_batches))
    return encrypted_df_hot, encrypted_batches

def trainGanModel(encrypted_batches):
    input_size = encrypted_batches[0].shape[1]
    tot_columns = input_size
    print('tot_columns:', tot_columns)
    #create the models
    generator = create_generator(input_size)
    discriminator = create_discriminator(input_size)
    #train GAN model
    gan = create_gan(discriminator,generator, input_size)
    train_gan(gan, generator, discriminator, encrypted_batches, tot_columns)
    return gan, generator
    
def saveGanModel(gan):
    folder_path = '../../resource/model/'
    filepath = folder_path + 'gan.keras'
    print('Saving the GAN Model')
    gan.save(folder_path  + 'gan.keras')
    print('Model saved at location:' , filepath)
    return filepath
    
def loadGanModel(filepath):   
    print('Loading the GAN Model')
    gan = load_model(filepath)
    
def generateSyntheticDataUsingGan(generator, columns):
    print("Generating Synthetic Data")
    noise = np.random.normal(0,1,[500,len(columns)])
    synthetic_data = generator.predict(noise)
    print('synthetic_data shape:',synthetic_data.shape)
    #generated_labels = discriminator.predict(noise)
    synthetic_data_df = pd.DataFrame(synthetic_data, columns=columns)
    print('synthetic_data_df shape:',synthetic_data_df.shape)
    # Calculate the mean for each column
    column_means = synthetic_data_df.mean()

    # Create a new DataFrame filled with 0s
    binary_gan_values_df = pd.DataFrame(0, index=synthetic_data_df.index, columns=synthetic_data_df.columns)
    
    # Use conditional statements to set values to 1 based on the mean
    for column in synthetic_data_df.columns:
        binary_gan_values_df[column] = synthetic_data_df[column].apply(lambda x: 1 if x >= column_means[column] else 0)
    print('binary_gan_values_df shape:',binary_gan_values_df.shape)
    return binary_gan_values_df

    