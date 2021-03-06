import numpy as np

from data_utilities import create_covariances
import functions_utilities as utils


def optimal_rotation(X, M):
    _, g_m = np.linalg.eigh(M)
    _, g_x = np.linalg.eigh(X)
    return np.einsum('...ij,...kj', g_m, g_x)



def optimal_reference_eigval(Sigmas):
    u = np.linalg.eigvalsh(Sigmas)
    return np.power(np.prod(u, axis=0), 1 / Sigmas.shape[0])



def optimal_reference_eigvec(Sigmas):
    _, vs = np.linalg.eigh(Sigmas)
    U, _, V = np.linalg.svd(np.sum(vs, axis=0))
    return np.einsum('...ij,...jk', U, V)


Number_of_Subjects = 20
Last_subject = Number_of_Subjects
Dimension_of_Covariances = 15

Sigmas = np.zeros(shape=(2 * Number_of_Subjects, Dimension_of_Covariances, Dimension_of_Covariances))
closed_eye = np.zeros(shape=(Number_of_Subjects, Dimension_of_Covariances, Dimension_of_Covariances))
opened_eye = np.zeros(shape=(Number_of_Subjects, Dimension_of_Covariances, Dimension_of_Covariances))

for i in range(Number_of_Subjects):
    subject = i + 1
    try:
        closed_eye[i], opened_eye[i] = create_covariances(subject)
    except FileNotFoundError:
        print('Subject {} does not exists')
        Last_subject = i
        break
    Sigmas[i] = opened_eye[i]
    Sigmas[Number_of_Subjects + i] = closed_eye[i]

ref_eigvals = optimal_reference_eigval(Sigmas)
ref_eigvect = optimal_reference_eigvec(Sigmas)
reference = np.einsum('...ij,...j,...kj', ref_eigvect, ref_eigvals, ref_eigvect)

Omegas = optimal_rotation(Sigmas, reference)

Sigmas_rotated = utils.rotate(Sigmas, Omegas)

