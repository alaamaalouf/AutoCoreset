import numpy as np
import sys, time, math

import utils



# Coreset taken from https://github.com/alaamaalouf/vector-summarization-coreset
def CaraIdxCoreset(P, u, dtype='float64'):
    while 1:
        n = np.count_nonzero(u)
        d = P.shape[1]
        u_non_zero = np.nonzero(u)
        if n <= d + 1: return P, u
        A = P[u_non_zero];
        reduced_vec = np.outer(A[0], np.ones(A.shape[0] - 1, dtype=dtype))
        A = A[1:].T - reduced_vec

        idx_of_try = 0;
        const = 10000;
        diff = np.infty;
        cond = sys.float_info.min

        _, _, V = np.linalg.svd(A, full_matrices=True)
        v = V[-1]
        diff = np.max(np.abs(np.dot(A, v)))
        v = np.insert(v, [0], -1 * np.sum(v))

        idx_good_alpha = np.nonzero(v > 0)
        alpha = np.min(u[u_non_zero][idx_good_alpha] / v[idx_good_alpha])

        w = np.zeros(u.shape[0], dtype=dtype)
        tmp = u[u_non_zero] - alpha * v
        tmp[np.argmin(tmp)] = 0.0
        w[u_non_zero] = tmp
        w[u_non_zero][np.argmin(w[u_non_zero])] = 0
        u = w

    return CaraIdxCoreset(P, w)


def updated_cara(P, w, coreset_size, dtype='float64'):
    start_time = time.time()
    d = P.shape[1];
    n = P.shape[0];
    m = 2 * d + 2;  # print (coreset_size,dtype)
    if n <= d + 1: return (P, w, np.array(list(range(0, P.shape[0]))))
    wconst = 1
    w_sum = np.sum(w)
    w = wconst * w / w_sum
    chunk_size = math.ceil(n / m)
    current_m = math.ceil(n / chunk_size)

    add_z = chunk_size - int(n % chunk_size)
    w = w.reshape(-1, 1)
    f = time.time()
    if add_z != chunk_size:
        zeros = np.zeros((add_z, P.shape[1]), dtype=dtype)
        P = np.concatenate((P, zeros))
        f3 = time.time();
        zeros = np.zeros((add_z, w.shape[1]), dtype=dtype)
        w = np.concatenate((w, zeros))

    idxarray = np.array(range(P.shape[0]))

    p_groups = P.reshape(current_m, chunk_size, P.shape[1])
    w_groups = w.reshape(current_m, chunk_size)
    idx_group = idxarray.reshape(current_m, chunk_size)
    w_nonzero = np.count_nonzero(w);
    counter = 1;  # print (w_nonzero, w)

    if not coreset_size: coreset_size = d + 1
    while w_nonzero > coreset_size:
        s0 = time.time()
        counter += 1
        groups_means = np.einsum('ijk,ij->ik', p_groups, w_groups)
        group_weigts = np.ones(groups_means.shape[0], dtype=dtype) * 1 / current_m

        Cara_p, Cara_w_idx = CaraIdxCoreset(groups_means, group_weigts, dtype=dtype)

        IDX = np.nonzero(Cara_w_idx)

        new_P = p_groups[IDX].reshape(-1, d)

        new_w = (current_m * w_groups[IDX] * Cara_w_idx[IDX][:, np.newaxis]).reshape(-1, 1)
        new_idx_array = idx_group[IDX].reshape(-1, 1)
        ##############################################################################3
        w_nonzero = np.count_nonzero(new_w)
        chunk_size = math.ceil(new_P.shape[0] / m)
        current_m = math.ceil(new_P.shape[0] / chunk_size)

        add_z = chunk_size - int(new_P.shape[0] % chunk_size)
        if add_z != chunk_size:
            new_P = np.concatenate((new_P, np.zeros((add_z, new_P.shape[1]), dtype=dtype)))
            new_w = np.concatenate((new_w, np.zeros((add_z, new_w.shape[1]), dtype=dtype)))
            new_idx_array = np.concatenate((new_idx_array, np.zeros((add_z, new_idx_array.shape[1]), dtype=dtype)))
        p_groups = new_P.reshape(current_m, chunk_size, new_P.shape[1])
        w_groups = new_w.reshape(current_m, chunk_size)
        idx_group = new_idx_array.reshape(current_m, chunk_size)
        ###########################################################

    return new_P, (w_sum * new_w / wconst).flatten(), time.time() - start_time,\
        (new_idx_array.reshape(-1).astype(int)).flatten()


def linregcoreset(P, w, b, c_size=None, is_svd=False, dtype='float64'):
    if not is_svd:
        P_tag = np.append(P, b, axis=1)
    else:
        P_tag = P
    n_tag = P_tag.shape[0]
    d_tag = P_tag.shape[1]
    P_tag = P_tag.reshape(n_tag, d_tag, 1)

    P_tag = np.einsum("ikj,ijk->ijk", P_tag, P_tag)
    P_tag = P_tag.reshape(n_tag, -1)
    n_tag = P_tag.shape[0]
    d_tag = P_tag.shape[1]  # print (w)

    coreset, coreset_weigts, new_idx_array = updated_cara(P_tag.reshape(n_tag, -1), w, c_size, dtype=dtype)

    if coreset is None:     return None, None, None
    coreset_weigts = coreset_weigts[(new_idx_array < P.shape[0])]
    new_idx_array = new_idx_array[(new_idx_array < P.shape[0])]

    # P_tag = np.append(P, b, axis=1)
    # coreset_tag = np.append(P[new_idx_array], b[new_idx_array], axis=1)
    if not is_svd:
        return P[new_idx_array], coreset_weigts.reshape(-1), b[new_idx_array]
    else:
        return P[new_idx_array], coreset_weigts.reshape(-1)


def medianOfMeans(P, m, k=100):
    mean_P = np.mean(P, axis=0)
    min_dist = np.inf
    chosen_opt = weights_opt = None
    for i in range(k):
        chosen, weights = generateUniformCoreset(P, None, w=np.ones((P.shape[0], )), m=m, replace=True,
                                                 return_indices=True)
        mean_coreset = np.average(P[chosen, :], weights=weights, axis=0)
        temp_dist = np.linalg.norm(mean_P - mean_coreset)
        if temp_dist < min_dist:
            chosen_opt = chosen
            weights_opt = weights
            min_dist = temp_dist

    return chosen_opt, weights_opt


def generateUniformCoreset(P, y, m, w, replace=False, return_indices=False, maintain_classes=False):
    n = P.shape[0]
    while(True):
        random_choice = np.random.choice(n, size=m, replace=replace)
        if y is not None and np.unique(y[random_choice], axis=0).shape[0] == np.unique(y, axis=0).shape[0]\
                or not maintain_classes:
            break
    chosen, counts = np.unique(random_choice, return_counts=True)
    weights = w[chosen] * counts * n/m
    if return_indices:
        return chosen, weights
    else:
        return P[chosen], (y[chosen] if y is not None else None), weights


def applyImportanceSamplingForOneMeanProblem(P, m, replace=True):
    sensitivity = np.linalg.norm(P, axis=1)
    t = np.sum(sensitivity)
    indices = np.random.choice(P.shape[0], p=sensitivity/t, size=m, replace=replace)
    chosen, counts = np.unique(indices, return_counts=True)

    return chosen, t * counts /sensitivity[chosen] / m




# def attainFWCoreset(P, chunk_idxs, sample_per_chunk):
#     if np.any(np.isnan(P)) or np.any(np.ifinf(P)):
#         raise ValueError('This is not supported! Please check your function and the chosen model')
#     # np.savez('data_before_falling.npz', P=P, chunk_idxs=chunk_idxs, sample_per_chunk=sample_per_chunk)
#     # s = time.time()
#     u = generateFrankWolfeCoreset(P, chunk_idxs.astype(np.int32), sample_per_chunk.astype(np.int32))
#     # print('Took {} seconds to compute FW using Cython'.format(time.time() - s))
#     #print('len(u) is {}, non-zero entries is {}'.format(len(u), np.count_nonzero(u)))
#     #print('u contains NaNs: {}'.format(np.any(np.isnan(u))))
#     if np.any(np.isnan(u)):
#         raise ValueError('This is not supported! Please check your function and the chosen model')
#     # if np.any(np.isnan(u))):
#
#     return np.nonzero(u)[0], u[u!=0]