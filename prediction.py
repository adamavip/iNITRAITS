from prepare_data import load_test_data
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow.keras.backend as K

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

# Load scaler
yscaler = joblib.load(open("./static/models/yscaler_Millet_Starch_Foss_2022-04-11.gz","rb"))

# Load the pretrained model
network = load_model("./static/models/Millet_DigestibleStarch_DL_HLR_2022-04-09")

def infer(data):
    preds = network.predict(data)
    K.clear_session()
    preds_org = yscaler.inverse_transform(preds)
    df = pd.DataFrame({"ID":list(range(1,len(preds)+1)), "Predictions":preds_org.flatten()})
    
    return df
