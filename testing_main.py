from testing import test
import pickle
from building_model import encoder
from creating_dist_mat import create_dist_matrix_for_encdoded
from CBIR import return_CBIR

def testing(date_time,epoch_number,encoder):
    # pickle_in = open("Data_split_pickles/train_val_test_split_20210107-162311.pickle","rb")
    pickle_in = open("Data_split_pickles/train_val_test_split_" + date_time +".pickle","rb")
    data_dic = pickle.load(pickle_in)

    X_train = data_dic['train'][0]
    pca_train = data_dic['train'][1]

    X_val = data_dic['val'][0]
    pca_val = data_dic['val'][1]

    X_test = data_dic['test'][0]
    pca_test = data_dic['test'][1]

    encoder.load_weights('Checkpoints/' + date_time + '/encoder-cp-'+epoch_number+'.ckpt')

    encoder, train_loss, val_loss, test_loss, test_preds = test(encoder,
                                                                X_train, X_val, X_test, pca_train, pca_val, pca_test,4)

    test_output_300 = test_preds[0]
    test_output_300_5 = test_preds[1]
    test_output_50 = test_preds[2]
    test_output_5 = test_preds[3]

    # encoder_500 = tf.keras.Model(inputs = encoder.inputs, outputs = encoder.layers[3].output)
    # encoder_300 = tf.keras.Model(inputs=encoder.inputs, outputs=encoder.layers[7].output)
    # encoder_50 = tf.keras.Model(inputs=encoder.inputs, outputs=encoder.layers[11].output)
    # encoder_5 = pass

    return train_loss,val_loss,test_loss,test_output_300,test_output_300_5,test_output_50,test_output_5,pca_test, X_test


if __name__=='__main__':
    train_loss,val_loss,test_loss,test_output_300, test_output_300_5, test_output_50, test_output_5, pca_test, pca_filepaths = testing('20210108-180603','0083',encoder)
    dist_300 = create_dist_matrix_for_encdoded(test_output_300)
    dist_300_5 = create_dist_matrix_for_encdoded(test_output_300_5)
    dist_50 = create_dist_matrix_for_encdoded(test_output_50)
    dist_5 = create_dist_matrix_for_encdoded(test_output_5)

    return_CBIR(12,dist_300,7,pca_filepaths)
    return_CBIR(12,dist_300_5,7,pca_filepaths)
    return_CBIR(12,dist_50,7,pca_filepaths)
    return_CBIR(12, dist_5, 7, pca_filepaths)

