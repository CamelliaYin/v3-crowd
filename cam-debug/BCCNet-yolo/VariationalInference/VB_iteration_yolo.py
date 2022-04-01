import scipy.special as ss
import numpy as np
import pdb


def VB_iteration(X, nn_output, alpha_volunteers, alpha0_volunteers):
    """
    performs one iteration of variational inference update for BCCNet (E-step)
    -- update for approximating posterior of true labels and confusion matrices
    I - number of data points
    M - number of true classes
    N - number of classes used by volunteers (normally M == N)
    K - number of volunteers (W)
    :param X: I X U X V X K volunteers answers, for image i, the grid choice u, the vth anchor box, the kth volunteer,
              -1 encodes a missing answer (where the volunteer can not identify abnormality there.)
    :param nn_output: (I x U X V) x M logits (not a softmax output!) note here the nn_output is only the partial output
                      from the object detection NN.
    :param alpha_volunteers: M X N X K - current parameters of posterior Dirichlet for confusion matrices
    :param alpha0_volunteers: M X N -  parameters of the prior Dirichlet for confusion matrix
    :return: q_t - approximating posterior for true labels, alpha_volunteers - updated posterior for confusion matrices,
        lower_bound_likelihood - ELBO
    """
    ElogPi_volunteer = expected_log_Dirichlet_parameters(alpha_volunteers)

    # q_t
    q_t, Njl, rho = expected_true_labels(X, nn_output, ElogPi_volunteer)

    # q_pi_workers
    alpha_volunteers = update_alpha_volunteers(alpha0_volunteers, Njl)

    # Low bound
    lower_bound_likelihood = compute_lower_bound_likelihood(alpha0_volunteers, alpha_volunteers, q_t, rho, nn_output)

    return q_t, alpha_volunteers, lower_bound_likelihood

# part of computing loss
def logB_from_Dirichlet_parameters(alpha):
    logB = np.sum(ss.gammaln(alpha)) - ss.gammaln(np.sum(alpha))

    return logB


def expected_log_Dirichlet_parameters(param):
    size = param.shape
    result = np.zeros_like(param)

    if len(size) == 1:
        result = ss.psi(param) - ss.psi(np.sum(param))
    elif len(size) == 2:  # when we take A_0 for everyone
        result = ss.psi(param) - np.transpose(np.tile(ss.psi(np.sum(param, 1)), (size[1], 1)))
    elif len(size) == 3:  # most of time for posterior cm
        for i in range(size[2]):
            result[:, :, i] = ss.psi(param[:, :, i]) - \
                              np.transpose(np.tile(ss.psi(np.sum(param[:, :, i], 1)), (size[1], 1)))
    else:
        raise Exception('param can have no more than 3 dimensions')

    return result


def expected_true_labels(X, nn_output, ElogPi_volunteer):
    I, U, K = X.shape  # I = no. of image, U = no. of anchor boxes in total, K = no. of volunteers
    M = ElogPi_volunteer.shape[0]  # M = Number of classes
    N = ElogPi_volunteer.shape[1]  # N = Number of classes used by volunteers

    rho = np.copy(nn_output)  # I x U x M logits
    # eq. 12:
    for k in range(K):
        inds = np.where(X[:, :, k] > -1)  # rule out missing values
        rho[inds[0], inds[1], :] = rho[inds[0], inds[1], :] + np.transpose(
            Elog[:, np.squeeze(X[inds[0], inds[1], k]), k])

    # normalisation: (minus the max of each anchor)
    rho = rho - np.transpose(np.tile(np.transpose(np.max(rho, 2)), (M, 1, 1)))

    # eq. 11:
    q_t = np.exp(rho) / np.maximum(1e-60, np.transpose(np.tile(np.transpose(np.sum(np.exp(rho), 2)), (M, 1, 1))))
    q_t = np.maximum(1e-60, q_t)

    # partial of eq. 8: (right side 2nd term)
    f_iu = np.zeros((M, N, K), dtype=np.float64)
    for k in range(K):
        for n in range(N):
            ids0 = np.where(X[:, :, k] == n)[0]
            ids1 = np.where(X[:, :, k] == n)[1]
            f_iu[:, n, k] = np.sum(q_t[ids0, ids1, :], 0)

    return q_t, f_iu, rho
# dim: (I x U x M), (M x N x K), (I x U x M)


# eq. 8:
def update_alpha_volunteers(alpha0_volunteers, f_iu):
    K = alpha0_volunteers.shape[2]
    alpha_volunteers = np.zeros_like(alpha0_volunteers)

    for k in range(K):
        alpha_volunteers[:, :, k] = alpha0_volunteers[:, :, k] + f_iu[:, :, k]

    return alpha_volunteers


def compute_lower_bound_likelihood(alpha0_volunteers, alpha_volunteers, q_t, rho, nn_output):
    K = alpha0_volunteers.shape[2]

    ll_pi_worker = 0
    for k in range(K):
        ll_pi_worker = ll_pi_worker - np.sum(logB_from_Dirichlet_parameters(alpha0_volunteers[:, :, w]) -
                                             logB_from_Dirichlet_parameters(alpha_volunteers[:, :, w]))

    ll_t = -np.sum(q_t * rho) + np.sum(np.log(np.sum(np.exp(rho), axis=2)))

    ll_nn = np.sum(q_t * nn_output) - np.sum(np.log(np.sum(np.exp(nn_output), axis=2)))

    ll = ll_pi_worker + ll_t + ll_nn  # VB lower bound

    return ll
