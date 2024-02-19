import sys
sys.path.append('C:/Users/chanc/BitsPythonClasses/Capstone Project/project/app/main/project/utils')

print(sys.path)
from utility import credit_utils


def printData():
    data = credit_utils.getCreditRiskData()
    print(credit_utils.get_encrypted_features(data))
    
    
if __name__ == '__main__':
    printData()