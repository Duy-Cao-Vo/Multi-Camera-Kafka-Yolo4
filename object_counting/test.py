import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
gpus = tf.config.experimental.list_physical_devices('GPU')
print('\nGPU: ',gpus,'\n')
tf.config.experimental.set_memory_growth(gpus[0], True)