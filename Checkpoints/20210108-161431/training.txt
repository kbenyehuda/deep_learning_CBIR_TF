import tensorflow as tf
import numpy as np
import random
import cv2
# from building_model import encoded,autoencoder,decoded
from creating_dist_mat import *
import datetime
import os
import pickle
from numba import prange
import pdb
from building_model import feature_extractor_layer


def return_label(num_pic):
    if num_pic < 707:
        return 1
    elif num_pic < 2134:
        return 2
    else:
        return 3


def create_inpt_data(inds, input_path, all_pca_feats):
    labels = []
    pca_feats = []
    inp_path_dir = 'D:/Users/Keren/Documents/university/Year 3/deep learning/my_project/DataBase/ProjectDB/DataBaseImage/'
    for ind in inds:
        idx = input_path[ind].index('Image/')
        inp_path = inp_path_dir + input_path[ind][idx + 6:]
        if ind == inds[0]:
            # x = np.array(cv2.imread(inp_path)[:, :, 0])[np.newaxis, :, :, np.newaxis]
            pic = cv2.resize(cv2.imread(inp_path), (224, 224), interpolation=cv2.INTER_AREA)
            x = np.array(pic)[np.newaxis, :, :, :]
        else:
            # pic = np.array(cv2.imread(inp_path)[:, :, 0])[np.newaxis, :, :, np.newaxis]
            pic = cv2.resize(cv2.imread(inp_path), (224, 224), interpolation=cv2.INTER_AREA)
            pic = np.array(pic)[np.newaxis, :, :, :]
            x = np.concatenate((x, pic))
        labels.append(return_label(ind))
        pca_feats.append(all_pca_feats[ind])
    # x = np.array(x)[:,:,:,np.newaxis] # here you have an issue
    pca_feats = np.array(pca_feats)
    # sift_dist_mat = create_squeezed_sift_matrix(big_dist_mat,inds)
    return x / 40, labels, pca_feats


def loss(encoder, x, labels, pca_feats):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    TF_output = feature_extractor_layer(x)
    encoded_300,encoded_300_5,encoded_50,encoded_5 = encoder(TF_output)
    # loss_value_1 = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(encoded_300-pca_feats)))
    loss_value_2 = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(encoded_300_5 - pca_feats[:,:5])))
    # loss_value_3 = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(encoded_50 - pca_feats[:,:50])))
    loss_value_4 = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(encoded_5 - pca_feats[:,:5])))
    # mse = tf.keras.losses.MeanSquaredError()
    # loss_value = mse(pca_feats, encoded)
    return loss_value_2 + loss_value_4, [encoded_300,encoded_300_5,encoded_50,encoded_5]


def grad(encoder, inputs, labels, pca_feats):
    with tf.GradientTape() as tape:
        loss_value, encoded = loss(encoder, inputs, labels, pca_feats)
    return loss_value, tape.gradient(loss_value, encoder.trainable_variables), encoded


def train(encoder, input_train_paths, input_val_paths, input_test_paths, pca_train, pca_val, pca_test,
          num_epochs, batch_size, num_batches, val_batch_size, verbose_range=50):
    ## saving the train val and test

    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)


    dic = {'train': [input_train_paths, pca_train], 'val': [input_val_paths, pca_val],
           'test': [input_test_paths, pca_test]}
    with open('Data_split_pickles/train_val_test_split_' + current_time + '.pickle', 'wb') as f:
        pickle.dump(dic, f)
    del dic

    # Keep results for plotting
    train_loss_results = []
    val_loss_results = []
    min_val_loss = np.inf
    grad_for_cur_epoch = dict()
    encoder_checkpoint_path = "Checkpoints/{}/encoder-cp-{epoch:04d}.ckpt"
    lr = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for epoch in range(num_epochs):
        print('epoch: ', epoch)
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_test_loss_avg = tf.keras.metrics.Mean()


        # if epoch == 1:
        #     lr = lr / 10
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  #lr = 0.001
        # if epoch == 7:
        #     lr = lr / 10    # lr = 0.0001
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # if epoch == 30:
        #     lr = lr / 10    # lr = 0.00001
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # if epoch == 40:
        #     lr = lr / 10    # lr = 0.000001
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # if epoch == 50:
        #     lr = lr / 10    # lr = 0.000001
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # Training loop - using batches of 32
        for batch_num in range(num_batches):
            # print('batch_num: ', batch_num)
            inds = random.sample(range(len(input_train_paths)), batch_size)
            x, labels, pca_feats = create_inpt_data(inds, input_train_paths, pca_train)

            # Optimize the model
            loss_value, grads, encoded = grad(encoder, x, labels, pca_feats)
            grad_for_cur_epoch[batch_num] = grads
            # print('mean encoded: ',encoded.numpy())
            # pdb.set_trace()
            zer_grads = True
            for i in range(len(grads)):
                if np.mean(grads[i]) != 0.0:
                    zer_grads = False
            if zer_grads:
                print('Got Zero Grads')
                raise ValueError

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current train batch loss
            # epoch_accuracy.update_state(y, model(x, training=True))

            if (batch_num + 1) * batch_size < len(input_val_paths):
                # Check on validation
                inds = random.sample(range(len(input_val_paths)), batch_size)
                x, labels, pca_feats = create_inpt_data(inds, input_val_paths, pca_val)
                val_loss, val_encoded = loss(encoder, x, labels, pca_feats)
                epoch_test_loss_avg.update_state(val_loss)  # Add current val batch loss

        vals = list(grad_for_cur_epoch.values())
        grads = list(np.array([sum(x) for x in zip(*vals)]) / batch_size)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables))

        if epoch_test_loss_avg.result() < min_val_loss:
            encoder.save_weights(encoder_checkpoint_path.format(current_time, epoch=epoch))
            min_val_loss = epoch_test_loss_avg.result()

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_test_loss_avg.result(), step=epoch)
        val_loss_results.append(epoch_test_loss_avg.result())

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_loss_avg.result(), step=epoch)
            # tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        # train_accuracy_results.append(epoch_accuracy.result())

        if epoch % verbose_range == 0:
            print("Epoch {:03d}: Train Loss: {:.3f}, Val Loss: {:.3f}".format(epoch,
                                                                              epoch_loss_avg.result(),
                                                                              val_loss))
    return encoder, train_loss_results, val_loss_results, grads