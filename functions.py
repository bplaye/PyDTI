
import os
import numpy as np
from collections import defaultdict


def load_data_from_file(dataset, folder):
    with open(os.path.join(folder, dataset + "_admat_dgc.txt"), "r") as inf:
        inf.next()
        int_array = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, dataset + "_simmat_dc.txt"), "r") as inf:  # the drug similarity file
        inf.next()
        drug_sim = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, dataset + "_simmat_dg.txt"), "r") as inf:  # the target similarity file
        inf.next()
        target_sim = [line.strip("\n").split()[1:] for line in inf]

    intMat = np.array(int_array, dtype=np.float64).T    # drug-target interaction matrix
    drugMat = np.array(drug_sim, dtype=np.float64)      # drug similarity matrix
    targetMat = np.array(target_sim, dtype=np.float64)  # target similarity matrix
    return intMat, drugMat, targetMat


def get_drugs_targets_names(dataset, folder):
    with open(os.path.join(folder, dataset + "_admat_dgc.txt"), "r") as inf:
        drugs = inf.next().strip("\n").split()
        targets = [line.strip("\n").split()[0] for line in inf]
    return drugs, targets


def cross_validation(intMat, seeds, cv=0, num=10, cv_type='fold'):
    cv_data = defaultdict(list)
    if cv_type == 'fold':
        for seed in seeds:
            num_drugs, num_targets = intMat.shape
            prng = np.random.RandomState(seed)
            if cv == 0:
                index = prng.permutation(num_drugs)
            if cv == 1:
                index = prng.permutation(intMat.size)
            step = index.size / num
            for i in xrange(num):
                if i < num - 1:
                    ii = index[i * step:(i + 1) * step]
                else:
                    ii = index[i * step:]
                if cv == 0:
                    test_data = np.array([[k, j] for k in ii for j in xrange(num_targets)], dtype=np.int32)
                elif cv == 1:
                    test_data = np.array([[k / num_targets, k % num_targets] for k in ii], dtype=np.int32)
                x, y = test_data[:, 0], test_data[:, 1]
                test_label = intMat[x, y]
                W = np.ones(intMat.shape)
                W[x, y] = 0
                cv_data[seed].append((W, test_data, test_label))
    elif cv_type == 'loo':
        cv_data = {seed: None for seed in seeds}
    elif cv_type == 'loo_balanced':
        pos_list = np.where(intMat == 1)
        neg_list = np.where(intMat == 0)
        for seed in seeds:
            choice = np.random.choice(neg_list[0].shape[0], size=pos_list[0].shape[0],
                                      replace=False)
            neg_list_local = (neg_list[0][choice], neg_list[1][choice])
            cv_data[seed] = [(pos_list[0][i], pos_list[1][i]) for i in range(len(pos_list[0]))] +\
                [(neg_list_local[0][i], neg_list_local[1][i])
                 for i in range(len(neg_list_local[0]))]
    return cv_data


def train(model, cv_data, intMat, drugMat, targetMat, cv_type='fold'):
    print(cv_type)
    if cv_type == 'fold':
        aupr, auc = [], []
        for seed in cv_data.keys():
            for W, test_data, test_label in cv_data[seed]:
                model.pred = np.full(intMat.shape, np.inf)
                model.fix_model(W, intMat, drugMat, targetMat, seed)
                aupr_val, auc_val = model.evaluation(test_data, test_label, intMat)
                aupr.append(aupr_val)
                auc.append(auc_val)
    elif cv_type == 'loo':
        seed = list(cv_data.keys())[0]
        model.pred = np.full(intMat.shape, np.inf)
        for i in range(intMat.shape[0]):
            for j in range(intMat.shape[1]):
                W = np.ones(intMat.shape)
                W[i, j], test_data, test_label = 0, np.asarray([[i, j]]), intMat[i, j]
                model.fix_model(W, intMat, drugMat, targetMat, seed)
                model.predict(test_data)
        aupr, auc = model.get_perf(intMat)
    elif cv_type == 'loo_balanced':
        aupr, auc = [], []
        for seed in cv_data.keys():
            model.pred = np.full(intMat.shape, np.inf)
            for i, j in cv_data[seed]:
                W = np.ones(intMat.shape)
                W[i, j], test_data, test_label = 0, np.asarray([[i, j]]), intMat[i, j]
                model.fix_model(W, intMat, drugMat, targetMat, seed)
                model.predict(test_data)
            aupr_val, auc_val = model.get_perf(intMat)
            aupr.append(aupr_val)
            auc.append(auc_val)

    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64)


def svd_init(M, num_factors):
    from scipy.linalg import svd
    U, s, V = svd(M, full_matrices=False)
    ii = np.argsort(s)[::-1][:num_factors]
    s1 = np.sqrt(np.diag(s[ii]))
    U0, V0 = U[:, ii].dot(s1), s1.dot(V[ii, :])
    return U0, V0.T


def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def write_metric_vector_to_file(auc_vec, file_name):
    np.savetxt(file_name, auc_vec, fmt='%.6f')


def load_metric_vector(file_name):
    return np.loadtxt(file_name, dtype=np.float64)
