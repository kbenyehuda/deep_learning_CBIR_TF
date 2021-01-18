import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_image(img_path):
    pic = cv2.imread(img_path)
    pic = np.squeeze(pic[:, :, 0] / np.max(pic))
    return pic

def return_label(img_idx):
    if int(img_idx) <= 708:
        return 1
    elif int(img_idx)<= 2134:
        return 2
    else:
        return 3

def return_best_idxs(image_idx,dist_mat,number_of_results):
    relevant_row = dist_mat[image_idx,:]
    return np.argsort(relevant_row)[:number_of_results]


def return_CBIR(image_idx,dist_mat,number_of_results,filepaths,return_idxs = False,show_results = True):
    idxs = return_best_idxs(image_idx,dist_mat,number_of_results)

    if show_results:
        show_results_func(filepaths,idxs[1:],image_idx)
    if return_idxs:
        return idxs[1:]


def show_results_func(filepaths,idxs,image_idx):
    inp_path_dir = 'D:/Users/Keren/Documents/university/Year 3/deep learning/my_project/DataBase/ProjectDB/DataBaseImage/'
    idx = filepaths[image_idx].index('Image/')
    first_img_path = inp_path_dir + filepaths[image_idx][idx + 6:]
    cur_pic = load_image(first_img_path)

    plt.figure()
    plt.subplot(1, len(idxs), 1)
    plt.imshow(cur_pic, cmap='gray')
    plt.title(filepaths[image_idx][idx + 6:-4] + 'label: ' + str(return_label(filepaths[image_idx][idx + 6:-4])))
    plt.xticks([])
    plt.yticks([])

    for i in range(len(idxs) - 1):
        idx = filepaths[idxs[i]].index('Image/')
        similar_img_path = inp_path_dir + filepaths[idxs[i]][idx + 6:]
        similar_pic = load_image(similar_img_path)
        plt.subplot(1, len(idxs), i + 2)
        plt.imshow(similar_pic, cmap='gray')
        plt.title(
            filepaths[idxs[i]][idx + 6:-4] + 'label: ' + str(return_label(filepaths[idxs[i]][idx + 6:-4])))
        plt.xticks([])
        plt.yticks([])
        plt.show()
