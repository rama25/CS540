from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    data_set = np.load(filename)
    mean_center = data_set - np.mean(data_set, axis=0)
    return mean_center


def get_eig_prop(S, prop):
    x_eigen, y_eigen = eigh(S, subset_by_value=[prop * np.trace(S), np.trace(S)])
    x_eigen_reverse = np.flip(x_eigen)
    y_eigh_reverse = np.flip(y_eigen, axis=1)
    diagnol_matrix = np.diag(x_eigen_reverse)
    return diagnol_matrix, y_eigh_reverse


def get_covariance(dataset):
    # discussed some aspects of the approach to the problem with Deepak Ranganathan and Ojal Sethi
    n_len = len(dataset)
    transposed_data_set = np.transpose(dataset)
    covariance_matrix = np.dot(transposed_data_set, dataset)
    covariance_value = covariance_matrix / n_len - 1
    return covariance_value


def project_image(image, U):
    u_transp = np.transpose(U)
    np_prod = np.dot(u_transp, image)
    np_proj = np.dot(np_prod, u_transp)
    return np_proj


def get_eig(S, m):
    x_eigh, y_eigh = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])
    x_eigen_reverse = np.flip(x_eigh)
    matrix_diagonol = np.diag(x_eigen_reverse)
    eigen_vec = np.flip(y_eigh, 1)
    return matrix_diagonol, eigen_vec


def display_image(orig, proj):
    original_img_reshape = np.transpose((orig.reshape(32, 32)))
    project_img_reshape = np.transpose((proj.reshape(32, 32)))
    figure, (p1, p2) = plt.subplots(1, 2)
    p1.set_title("Original")
    p2.set_title("Projection")
    fig1 = p1.imshow(original_img_reshape, aspect="equal")
    figure.colorbar(fig1, ax=p1)
    fig2 = p2.imshow(project_img_reshape, aspect="equal")
    figure.colorbar(fig2, ax=p2)
    plt.show()
