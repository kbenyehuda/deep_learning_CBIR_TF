# from dataset import *
from training import *
# from building_model import *
def test(encoder, input_train_paths, input_val_paths, input_test_paths, pca_train, pca_val, pca_test,
         batch_size, verbose_range=50):
    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()

    for inp_type in range(3):
        print('started input type ',inp_type)
        # for inp_type in range(3):
        if inp_type == 0:
            inpt = input_train_paths
            pca_inpt = pca_train
        elif inp_type == 1:
            inpt = input_val_paths
            pca_inpt = pca_val
        else:
            inpt = input_test_paths
            pca_inpt = pca_test

        num_batches = np.floor(len(inpt) / batch_size).astype('int32') + 1
        for batch_num in range(num_batches):

            if batch_num%verbose_range==0:
                print('batch ',batch_num,' out of ',num_batches)
            if batch_num == num_batches - 1:
                inds = np.linspace(batch_num*batch_size, len(inpt) - 1,len(inpt) - batch_num*batch_size).astype('int32')
            else:
                inds = np.linspace(batch_num * batch_size, (batch_num + 1) * batch_size - 1, batch_size).astype('int32')
            x, labels, pca_feats = create_inpt_data(inds, inpt, pca_inpt)

            loss_val, encoded = loss(encoder, x, labels, pca_feats)

            if inp_type == 0:
                train_loss.update_state(loss_val)
            elif inp_type == 1:
                val_loss.update_state(loss_val)
            else:
                test_loss.update_state(loss_val)
                if batch_num == 0:
                    # test_preds = encoded.numpy()
                    test_preds_300 = encoded[0][:].numpy()
                    test_preds_300_5 = encoded[1][:].numpy()
                    test_preds_50 = encoded[:][2].numpy()
                    test_preds_5 = encoded[:][3].numpy()
                else:
                    test_preds_300 = np.concatenate((test_preds_300, encoded[0][:].numpy()))
                    test_preds_300_5 = np.concatenate((test_preds_300_5, encoded[1][:].numpy()))
                    test_preds_50 = np.concatenate((test_preds_50, encoded[2][:].numpy()))
                    test_preds_5 = np.concatenate((test_preds_5, encoded[3][:].numpy()))
    return encoder, train_loss.result(), val_loss.result(), test_loss.result(), [test_preds_300,test_preds_300_5,test_preds_50,test_preds_5]
    # return encoder,val_loss.result(),test_loss.result(),test_preds