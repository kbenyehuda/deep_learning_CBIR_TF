from glob import glob
import numpy as np
import pickle
from training import return_label
from sklearn.model_selection import train_test_split

# base_pic_path = '/content/drive/MyDrive/Image processing 3?/Project/'
# all_filepaths = glob(base_pic_path+ 'ProjectDB/DataBaseImage/*')
# all_filepaths.pop(707)
# print('got filepaths')


# base_pca_path = '/content/drive/MyDrive/Image processing 3?/Project/deep_learning_network_1/'
base_pca_path = 'D:/Users/Keren/Documents/university/Year 3/deep learning/'
PCA_feats_path = base_pca_path + 'inpt_data.pickle'
pickle_in = open(PCA_feats_path, 'rb')
dic = pickle.load(pickle_in)
all_pca_feats = dic['all_pca_feats']
all_filepaths = dic['all_filepaths']

'''
NORMALIZING THE PCA FEATS IN A MAYBE BETTER WAY
'''


def normalize(x):
    # z = (x - np.mean(x)) / np.std(x)
    z = (x - np.min(x))/(np.max(x) - np.min(x))
    return z


new_pca_feats = np.zeros((np.shape(all_pca_feats)[0], 300))
for i in range(300):
    new_pca_feats[:, i] = normalize(all_pca_feats[:, i])

all_pca_feats = new_pca_feats
# pickle_in = open(base_pca_path + "all_pics_matrix.pickle","rb")
# all_pics = pickle.load(pickle_in)
print('got PCA feats, filepaths and pics')

# labels = np.squeeze(np.array(labels))

# print(np.shape(labels))
# print(np.shape(all_filepaths))

# import cv2
# bad_inds = []
# for i in range(len(all_filepaths)):
#     if np.shape(cv2.imread(all_filepaths[i]))!=(512,512,3):
#         # print('i ',i,' is bad')
#         # print('it has shape: ',np.shape(cv2.imread(all_filepaths[i])))
#         bad_inds.append(i)
# print('got bad inds')

# all_pca_feats = np.delete(all_pca_feats,np.array(bad_inds),axis=0)
# all_filepaths = np.delete(all_filepaths,np.array(bad_inds))
# print('deleted bad inds')
X_train, X_rest, pca_train, pca_rest = train_test_split(all_filepaths, all_pca_feats, test_size=0.3, random_state=1)

X_val, X_test, pca_val, pca_test = train_test_split(X_rest, pca_rest, test_size=0.5, random_state=1)
print('finished splitting up the data')
del X_rest
del pca_rest

# pickle_in = open("distance_matrix.pickle","rb")
# dic = pickle.load(pickle_in)
# big_dist_mat = dic['distance_matrix']

