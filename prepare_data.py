import pandas as pd
import numpy as np
from preprocessing import preprocess
import joblib


# Load scaler
xscaler = joblib.load(open("./static/models/xscaler_Millet_Starch_Foss_2022-04-11.gz","rb"))

def load_test_data(test_fname, start_band='1350.155463'):
    

    test = pd.read_csv(test_fname, sep=";")

    #Build predictors
    
    x_test = test.loc[:,start_band:]

    #Convert to numpy arrays
    x_test = x_test.to_numpy()

    #Apply preprocessing operations
    x_test = preprocess(x_test)

    #Minmax Normalization of x and y
    x_test = xscaler.transform(x_test)

    # Reshaping arrays
    x_test = np.expand_dims(x_test, axis=2)


    return x_test
