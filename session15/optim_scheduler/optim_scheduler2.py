

import tensorflow as tf
import numpy as np
BATCH_SIZE = 512
batches_per_epoch = 50000//BATCH_SIZE + 1
last_batch_size = 50000 % BATCH_SIZE
EPOCHS=24
LEARNING_RATE = 0.4 
WEIGHT_DECAY = 5e-4 
MOMENTUM = 0.9 

lr_schedule = lambda t: np.interp([t], [0, int((EPOCHS+1)*0.2), int((EPOCHS+1)*0.7), EPOCHS], [LEARNING_RATE/5.0, LEARNING_RATE, LEARNING_RATE/5.0, 0])[0]
global_step = tf.train.get_or_create_global_step()
lr_func = lambda: lr_schedule(global_step/batches_per_epoch)/BATCH_SIZE
opt = tf.train.MomentumOptimizer(lr_func, momentum=MOMENTUM, use_nesterov=True)