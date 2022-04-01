#  Copyright (c) 2019. University of Oxford

import numpy as np
# def of A_0

# the idea is that, for each individual, we set the same prior parameters A_0.
# if take alpha_diag_prior = 0.1 (Olga's experiment setting)
## sim_prior = initialise_prior(3,2,0.1)
## print(sim_prior[:,:,1]) == print(sim_prior[:,:,1])
## [[1.1 1.  1. ]
## [1.  1.1 1. ]
## [1.  1.  1.1]]


def initialise_prior(n_classes, n_volunteers, alpha_diag_prior):
    """
    Create confusion matrix prior for every volunteer - the same prior for each volunteer
    :param n_classes: number of classes (int)
    :param n_volunteers: number of crowd members (int)
    :param alpha_diag_prior: prior for confusion matrices is assuming reasonable crowd members with weak dominance of a
    diagonal elements of confusion matrices, i.e. prior for a confusion matrix is a matrix of all ones where
    alpha_diag_prior is added to diagonal elements (float)
    ### a matrix full of ones, plus alpha_diag_prior is added to diagonal elements (float)
    :return: numpy nd-array of the size (n_classes, n_classes, n_volunteers)
    """
    alpha_volunteer_template = np.ones((n_classes, n_classes), dtype=np.float64) + alpha_diag_prior * np.eye(n_classes)
    ### as commented above.
    return np.tile(np.expand_dims(alpha_volunteer_template, axis=2), (1, 1, n_volunteers))
    ### expand the dim to (n_classes, n_classes, 1), then repeat it to dimension (1, 1, n_volunteers)

    ### so my prior would be the same for each volunteer, and for each silce i,
    ### (n_classes, n_classes, i) would be full of 1, only diagonal entries are 1+alpha_diag_prior
    ### where alpha_diag_prior is a float





