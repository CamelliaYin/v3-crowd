import tensorflow as tf
import os
import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt


from NNArchitecture.lenet5_mnist import cnn_for_mnist
from SyntheticCrowdsourcing.synthetic_crowd_volunteers import generate_volunteer_labels
# from VariationalInference.VB_iteration import VB_iteration
from utils.utils_dataset_processing import shrink_arrays
from VariationalInference import confusion_matrix


rseed = 1000
np.random.seed(rseed)
#tf.set_random_seed(rseed)

# parameters
n_classes = 10
crowdsourced_labelled_train_data_ratio = 0.5
n_crowd_members = 4
crowd_member_reliability_level = 0.6
confusion_matrix_diagonal_prior = 1e-1
n_epoch = 100
batch_size = 32
convergence_threshold = 1e-6

# load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=os.getcwd() + '/mnist.npz')

# expand images for a cnn
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# select subsample of train data to be "labelled" by crowd members
labelled_train, whole_train = shrink_arrays([x_train, y_train], crowdsourced_labelled_train_data_ratio, is_shuffle=True)
x_labelled_train = labelled_train[0]
y_labelled_train = labelled_train[1]
x_train = whole_train[0]
y_train = whole_train[1]

# generate synthetic crowdsourced labels
crowdsourced_labels = generate_volunteer_labels(n_volunteers=n_crowd_members, n_classes=n_classes, gt_labels=y_labelled_train,
                                                n_total_tasks=x_train.shape[0],
                                                reliability_level=crowd_member_reliability_level)

# set up a neural net
cnn_model = cnn_for_mnist()

# set up variational parameters
prior_param_confusion_matrices = confusion_matrix.initialise_prior(n_classes=n_classes, n_volunteers=n_crowd_members,
                                                                   alpha_diag_prior=confusion_matrix_diagonal_prior)
variational_param_confusion_matrices = np.copy(prior_param_confusion_matrices)
# shape (10, 10, 4)

## ititally the \alpha_0 is the same for every volunteer, with 1.1 on the diagonal, and 1 for the rest.
## for sure, initally we have \alpha_0 and \alpha the same

# initial variational inference iteration (initialisation of approximating posterior of true labels)
initial_nn_output_for_vb_update = np.random.randn(x_train.shape[0], n_classes)
## (60000, 10)
## for each object we have 10 logits coresponding to 10 classes




def logB_from_Dirichlet_parameters(alpha):
    logB = np.sum(ss.gammaln(alpha)) - ss.gammaln(np.sum(alpha))

    return logB
###  ss.gammaln(): Logarithm of the absolute value of the gamma function.

logB = logB_from_Dirichlet_parameters(prior_param_confusion_matrices)
# logB is a single number

def expected_log_Dirichlet_parameters(param): #Eq.(16)
    # param: alpha_volunteers
    size = param.shape
    # shape:(J X L X W)

    result = np.zeros_like(param)

    if len(size) == 1:
        result = ss.psi(param) - ss.psi(np.sum(param))
        # ss.psi(): The logarithmic derivative of the gamma function evaluated at param
    elif len(size) == 2:
        result = ss.psi(param) - np.transpose(np.tile(ss.psi(np.sum(param, 1)), (size[1], 1)))
    elif len(size) == 3:
        for i in range(size[2]): # for each volunteer
            result[:, :, i] = ss.psi(param[:, :, i]) - \
                              np.transpose(np.tile(ss.psi(np.sum(param[:, :, i], 1)), (size[1], 1)))
    else:
        raise Exception('param can have no more than 3 dimensions')

    return result

result = expected_log_Dirichlet_parameters(prior_param_confusion_matrices)
# result is in shape (10, 10, 4)

### Confusion: for len(size)==3, the dim of ss.psi(param[:, :, i]) is J x L
###            the dim of np.sum is J x 1 (add the whole row)
###           the dim of np.tile is is J^2 x 1, and after transpose we have 1 x J^2
###            can't deduct given the dim doesn't match

def expected_true_labels(X, nn_output, ElogPi_volunteer):
    N, W = X.shape  # N = Number of subjects, W = Number of volunteers.
    J = ElogPi_volunteer.shape[0]  # J = Number of classes
    L = ElogPi_volunteer.shape[1] # L = Number of classes used by volunteers
### comments: ElogPi_volunteer is the LHS of Eq. (16), the A_ji^(k)
    rho = np.copy(nn_output) # N X J logits

    for w in range(W):
        inds = np.where(X[:, w] > -1) #extract index for filled objects
        rho[inds, :] = rho[inds, :] + np.transpose(ElogPi_volunteer[:, np.squeeze(X[inds, w]), w])
# rho[inds, :] pick out rows/objects logits that are labelled  that are labelled
    rho = rho - np.transpose(np.tile(np.max(rho, 1), (J, 1)))

    q_t = np.exp(rho) / np.maximum(1e-60, np.transpose(np.tile(np.sum(np.exp(rho), 1), (J, 1))))
    q_t = np.maximum(1e-60, q_t)

    Njl = np.zeros((J, L, W), dtype=np.float64)
    for w in range(W):
        for l in range(L):
            inds = np.where(X[:, w] == l)[0]
            Njl[:, l, w] = np.sum(q_t[inds, :], 0)

    return q_t, Njl, rho

q_t, Njl, rho = expected_true_labels(crowdsourced_labels, initial_nn_output_for_vb_update, result)

# ElogPi_volunteer = result


def update_alpha_volunteers(alpha0_volunteers, Njl):
    W = alpha0_volunteers.shape[2]
    alpha_volunteers = np.zeros_like(alpha0_volunteers)

    for w in range(W):
        alpha_volunteers[:, :, w] = alpha0_volunteers[:, :, w] + Njl[:, :, w]

    return alpha_volunteers

alpha_volunteers = update_alpha_volunteers(prior_param_confusion_matrices, Njl)


def compute_lower_bound_likelihood(alpha0_volunteers, alpha_volunteers, q_t, rho, nn_output):
    W = alpha0_volunteers.shape[2]

    ll_pi_worker = 0
    for w in range(W):
        ll_pi_worker = ll_pi_worker - np.sum(logB_from_Dirichlet_parameters(alpha0_volunteers[:, :, w]) -
                                             logB_from_Dirichlet_parameters(alpha_volunteers[:, :, w]))

    ll_t = -np.sum(q_t * rho) + np.sum(np.log(np.sum(np.exp(rho), axis=1)), axis=0)

    ll_nn = np.sum(q_t * nn_output) - np.sum(np.log(np.sum(np.exp(nn_output), axis=1)), axis=0)

    ll = ll_pi_worker + ll_t + ll_nn  # VB lower bound

    return ll



