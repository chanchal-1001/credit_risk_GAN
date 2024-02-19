###Project Name : USER DATA PROTECTION IN MACHINE LEARNING MODELS
This project presents a methodology to credit risk assessment in banking that ensures privacy preservation using a combination of Generative Adversarial Network (GANs), Deterministic Encryption and Binary Cross Entropy classifier.


#Table of Contents

#Introduction
Banks handle sensitive personal data, and it's crucial to ensure data privacy even when the data is used for machine learning tasks. This project will involve implementing a machine learning model for credit risk assessment while preserving the privacy of user data

#Features
In this project, we used different deep learning approaches like deep neural networks, generative adversarial networks and privacy preserving data encryption techniques.
This project work highlights that homomorphic encryption approaches such as CKKS and Paillier encryption can be used for simple mathematical tasks but cannot be used with deep learning algorithms and this is an area of active research.
We were able to create a simple setup where credit data can be converted into an encrypted form and when fed into a neural network, we were able to classify good and bad credit.
On top of it, we were able to create GAN which was trained to generate synthetic data. In case of class imbalance issues, clients should be able to generate synthetic data which will improve overall training and prediction.

#Installation
install python > 3.10
pip install tensorflow
pip install torch


#Usage
 http://127.0.0.1:8049/predictCreditRisk is expose and can be run using python api.py command 
 Send input Data to api is in below format:
 [
    {
        "existingchecking": "A11",
        "duration": 6,
        "credithistory": "A34",
        "purpose": "A43",
        "creditamount": 1169,
        "savings": "A65",
        "employmentsince": "A75",
        "installmentrate": 4,
        "statussex": "A93",
        "otherdebtors": "A101",
        "residencesince": 4,
        "property": "A121",
        "age": 67,
        "otherinstallmentplans": "A143",
        "housing": "A152",
        "existingcredits": 2,
        "job": "A173",
        "peopleliable": 1,
        "telephone": "A192",
        "foreignworker": "A201"
    }
]
#Training
 Run python main.py command under services folder. This will train and save the model under resources/model folder.
#Results
 - Result will be 1 or 2 values, 1 for Good Credit and 2 for Bad Credit.
[
    [
        2
    ],
    [
        1
    ],
    [
        2
    ],
    [
        1
    ],
    [
        1
    ]
]
#Contributing
We had created GANs which could still be trained and evolved over larger data set to generate real looking data. Thus, GANs can further be explored to improve classification capacity of the binary classifier.

