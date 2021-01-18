import tensorflow as tf
from building_model import *
from training import train
from building_model import *

num_epochs = 2
batch_size = 1
num_batches = 3
val_batch_size = 1

hist = train(encoder,decoder,
             X_train,X_val,X_test,
             labels_train,labels_val,labels_test,
             num_epochs,batch_size,num_batches,val_batch_size)
