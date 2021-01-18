'''

This project is decoding an image to it's 50 PCA SIFT features

'''

from dataset import *
from training import train
from building_model import *

num_epochs = 400
batch_size = 32
num_batches = 10
val_batch_size = 32

encoder, train_loss_results, val_loss_results,last_grads = train(encoder,
                                                      X_train,X_val,X_test,
                                                      pca_train,pca_val,pca_test,
                                                      num_epochs,batch_size,num_batches,val_batch_size,verbose_range = 1)
