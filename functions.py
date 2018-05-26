import os
import numpy as np
from collections import defaultdict
from sklearn import cluster
import time


def get_name_method(method):
    if method == 'wnngip':
      m = 'RLSWNN'
    elif method == 'gip':
      m = 'RLS'
    elif method == 'nngip':
      m = 'NNRLS'
    elif method == 'nnwnngip':
      m = 'NNRLSWNN'
    elif method == 'nrlmf':
      m = 'NRLMF'
    elif method == 'nnkronsvm':
      m = 'NNKronSVM'
    elif method == 'nnkronsvmgip':
      m = 'NNKronSVMGIP'
    elif method == 'nnkronwnnsvmgip':
      m = 'NNKronWNNSVMGIP'
    elif method == 'nnkronwnnsvm':
      m = 'NNKronWNNSVM'
    return m


def load_data_from_file(dataset, folder):
    if 'drugbank' in dataset:
        if '2000' in dataset:
            limit = 2000
        elif '5000' in dataset:
            limit = 5000
        else:
            limit = None
        dataset = 'DrugBank'
    else:
        limit = None
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

    if False:
        n_mol, n_prot, n_mol_orphan, n_prot_orphan = intMat.shape[0], intMat.shape[1], 0, 0
        for i in range(intMat.shape[0]):
            if len(np.where(intMat[i, :] == 1)[0]) == 1:
                n_mol_orphan += 1
        for j in range(intMat.shape[1]):
            if len(np.where(intMat[:, j] == 1)[0]) == 1:
                n_prot_orphan += 1
        n_interaction = len(np.where(intMat == 1)[0])
        print(dataset, n_mol, n_prot, n_mol_orphan, n_prot_orphan, n_interaction)
        exit(1)

    drugMat = np.array(drug_sim, dtype=np.float64)      # drug similarity matrix
    targetMat = np.array(target_sim, dtype=np.float64)  # target similarity matrix
    return intMat, drugMat, targetMat, limit


def get_drugs_targets_names(dataset, folder):
    if 'drugbank' in dataset:
        dataset = 'DrugBank'
    with open(os.path.join(folder, dataset + "_admat_dgc.txt"), "r") as inf:
        drugs = inf.next().strip("\n").split()
        targets = [line.strip("\n").split()[0] for line in inf]
    return drugs, targets


def make_hierarchical_clustering(samples, D, T, num):
    dist = np.zeros((len(samples[0]), len(samples[0])))
    for i1 in range(len(samples[0])):
        for i2 in range(i1, len(samples[0])):
            dist[i1, i2] = \
                (1 - D[samples[0][i1], samples[0][i2]] * T[samples[1][i1], samples[1][i2]])
            dist[i2, i1] = dist[i1, i2]
    clust = cluster.AgglomerativeClustering(num, affinity='precomputed', linkage='complete')
    clust = clust.fit_predict(dist)
    list_cluster = [[[], []] for _ in range(num)]
    for isample in range(len(samples[0])):
        list_cluster[clust[isample]][0].append(samples[0][isample])
        list_cluster[clust[isample]][1].append(samples[1][isample])
    return list_cluster


def order_neg_and_pos_cluster(list_cluster_pos, list_cluster_neg, D, T):
    list_dist = [[] for _ in range(len(list_cluster_pos))]
    for i1 in range(len(list_cluster_pos)):
      for i2 in range(len(list_cluster_neg)):
        v = 0
        for i1s in range(len(list_cluster_pos[i1][0])):
          for i2s in range(len(list_cluster_neg[i2][0])):
            v += (1 - D[list_cluster_pos[i1][0][i1s], list_cluster_neg[i2][0][i2s]] *
                  T[list_cluster_pos[i1][1][i1s], list_cluster_neg[i2][1][i2s]])
        list_dist[i1].append(v)
    ii_pos, ii_neg = [], []
    while len(ii_pos) != len(list_cluster_pos):
      min_ = 10000000
      i1min, i2min = None, None
      for i1 in range(len(list_cluster_pos)):
        for i2 in range(len(list_cluster_neg)):
          if list_dist[i1][i2] < min_:
            i1min, i2min = i1, i2
            min_ = list_dist[i1][i2]
      ii_pos.append(list_cluster_pos[i1min])
      ii_neg.append(list_cluster_neg[i2min])
      for i1 in range(len(list_cluster_pos)):
        for i2 in range(len(list_cluster_neg)):
          if i1 == i1min or i2 == i2min:
            list_dist[i1][i2] = 10000000
    return ii_pos, ii_neg


def cross_validation(intMat, D, T, seeds, cv=0, limit=None, num=10, cv_type='fold'):
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
    elif cv_type == 'fold_balanced':
        pos_list = np.where(intMat == 1)
        if limit is not None:
            choice = np.random.choice(pos_list[0].shape[0], size=limit, replace=False)
            pos_list = (pos_list[0][choice], pos_list[1][choice])
        neg_list = np.where(intMat == 0)
        for seed in seeds:
            np.random.seed(seed)
            choice = np.random.choice(neg_list[0].shape[0], size=pos_list[0].shape[0],
                                      replace=False, )
            neg_list_ = np.asarray((neg_list[0][choice], neg_list[1][choice]))

            prng = np.random.RandomState(seed)
            index = prng.permutation(pos_list[0].shape[0])
            step = pos_list[0].shape[0] / num
            for i in xrange(num):
                if i < num - 1:
                    ii = index[i * step:(i + 1) * step]
                else:
                    ii = index[i * step:]
                test_data_pos = np.array([[pos_list[0][k], pos_list[1][k]] for k in ii],
                                         dtype=np.int32)
                test_data_neg = np.array([[neg_list_[0][k], neg_list_[1][k]] for k in ii],
                                         dtype=np.int32)
                test_data = np.concatenate((test_data_pos, test_data_neg), axis=0)

                x, y = test_data[:, 0], test_data[:, 1]
                test_label = intMat[x, y]
                W = np.ones(intMat.shape)
                W[x, y] = 0
                cv_data[seed].append((W, test_data, test_label))
    elif cv_type == 'cluster_fold_balanced':
        pos_list = np.where(intMat == 1)
        if limit is not None:
            choice = np.random.choice(pos_list[0].shape[0], size=limit, replace=False)
            pos_list = (pos_list[0][choice], pos_list[1][choice])
        list_cluster_pos = make_hierarchical_clustering(pos_list, D, T, num)
        neg_list = np.where(intMat == 0)
        for seed in seeds:
            np.random.seed(seed)
            choice = np.random.choice(neg_list[0].shape[0], size=pos_list[0].shape[0],
                                      replace=False)
            neg_list_ = np.asarray((neg_list[0][choice], neg_list[1][choice]))

            list_cluster_neg = make_hierarchical_clustering(neg_list_, D, T, num)
            ii_pos, ii_neg = order_neg_and_pos_cluster(list_cluster_pos, list_cluster_neg, D, T)
            for i in xrange(num):
                ii_p, ii_n = ii_pos[i], ii_neg[i]
                test_data_pos = np.array([[ii_p[0][k], ii_p[1][k]] for k in range(len(ii_p[0]))],
                                         dtype=np.int32)
                test_data_neg = np.array([[ii_n[0][k], ii_n[1][k]] for k in range(len(ii_n[0]))],
                                         dtype=np.int32)
                test_data = np.concatenate((test_data_pos, test_data_neg), axis=0)

                x, y = test_data[:, 0], test_data[:, 1]
                test_label = intMat[x, y]
                W = np.ones(intMat.shape)
                W[x, y] = 0
                cv_data[seed].append((W, test_data, test_label))

    elif cv_type == 'loo':
        cv_data = {seed: None for seed in seeds}
    elif cv_type == 'loo_balanced' or cv_type == 'hard_loo_balanced' or \
            cv_type == 'loo_balanced_no_orphans':
        pos_list = np.where(intMat == 1)
        print('limit', limit)
        if limit is not None:

            choice = np.random.choice(pos_list[0].shape[0], size=limit, replace=False)
            pos_list = (pos_list[0][choice], pos_list[1][choice])
        print('nb pos samples', len(pos_list[0]))
        neg_list = np.where(intMat == 0)
        for seed in seeds:
            np.random.seed(seed)
            choice = np.random.choice(neg_list[0].shape[0], size=pos_list[0].shape[0],
                                      replace=False)
            neg_list_local = (neg_list[0][choice], neg_list[1][choice])
            cv_data[seed] = [(pos_list[0][i], pos_list[1][i]) for i in range(len(pos_list[0]))] +\
                [(neg_list_local[0][i], neg_list_local[1][i])
                 for i in range(len(neg_list_local[0]))]

    return cv_data


def train_single(model, method, dataset, cv_data, intMat, drugMat, targetMat, cv_type, i_test):
    if cv_type in ['fold', 'fold_balanced', 'cluster_fold_balanced']:
        print(['fold', 'fold_balanced', 'cluster_fold_balanced'])
        list_test = []
        i = 0
        for seed in cv_data.keys():
            for W, test_data, test_label in cv_data[seed]:
                if i == i_test:
                    R = W * intMat
                    model.pred = np.full(intMat.shape, np.inf)
                    model.fix_model(W, intMat, drugMat, targetMat, seed)
                    aupr_val, auc_val, pred = model.evaluation(test_data, test_label, intMat, R)
                i += 1
        return aupr_val, auc_val, pred, test_label
    # elif cv_type == 'loo':
    #     seed = list(cv_data.keys())[0]
    #     model.pred = np.full(intMat.shape, np.inf)
    #     for i in range(intMat.shape[0]):
    #         for j in range(intMat.shape[1]):
    #             W = np.ones(intMat.shape)
    #             W[i, j], test_data, test_label = 0, np.asarray([[i, j]]), intMat[i, j]
    #             model.fix_model(W, intMat, drugMat, targetMat, seed)
    #             model.predict(test_data)
    #     aupr, auc = model.get_perf(intMat)
    #     return None, None, pred, test
    elif cv_type == 'loo_balanced':
        aupr, auc = [], []
        iii = 0
        for seed in cv_data.keys():
          if iii == i_test:
            model.pred = np.full(intMat.shape, np.inf)
            for i, j in cv_data[seed]:
                W = np.ones(intMat.shape)
                W[i, j], test_data, test_label = 0, np.asarray([[i, j]]), intMat[i, j]
                R = W * intMat
                model.fix_model(W, intMat, drugMat, targetMat, seed)
                model.predict(test_data, R)
          iii += 1
        pred_ind = np.where(model.pred != np.inf)
        pred_local = model.pred[pred_ind[0], pred_ind[1]]
        test_local = intMat[pred_ind[0], pred_ind[1]]
        aupr_val, auc_val = model.get_perf(intMat)
        return aupr_val, auc_val, pred_local, test_local

    elif cv_type == 'loo_balanced_no_orphans':
        aupr, auc = [], []
        iii = 0
        for seed in cv_data.keys():
          if iii == i_test:
            model.pred = np.full(intMat.shape, np.inf)
            for i, j in cv_data[seed]:
                W = np.ones(intMat.shape)
                W[i, j], test_data, test_label = 0, np.asarray([[i, j]]), intMat[i, j]
                if len(np.where(intMat[i, :] == 1)[0]) != 0 and \
                        len(np.where(intMat[:, j] == 1)[0]) != 0:
                    R = W * intMat
                    model.fix_model(W, intMat, drugMat, targetMat, seed)
                    model.predict(test_data, R)
          iii += 1
        pred_ind = np.where(model.pred != np.inf)
        pred_local = model.pred[pred_ind[0], pred_ind[1]]
        test_local = intMat[pred_ind[0], pred_ind[1]]
        aupr_val, auc_val = model.get_perf(intMat)
        return aupr_val, auc_val, pred_local, test_local

    elif cv_type == 'hard_loo_balanced':
        list_test = []
        iii = 0
        for seed in cv_data.keys():
          if iii == i_test:
            model.pred = np.full(intMat.shape, np.inf)
            for n, (i, j) in enumerate(cv_data[seed]):
              t1 = time.time()
              W = np.ones(intMat.shape)
              W[i, j], test_data, test_label = 0, np.asarray([[i, j]]), intMat[i, j]
              # print('np.where(intMat[i, :] == 1)[0]', np.where(intMat[i, :] == 1)[0])
              # print('W[np.where(intMat[:, j] == 1)[0], j]', W[np.where(intMat[:, j] == 1)[0], j])
              # W[i, np.where(intMat[i, :] == 1)[0]] = 0
              # W[np.where(intMat[:, j] == 1)[0], j] = 0
              W[i, :].fill(0)
              W[:, j].fill(0)
              if 'drugbank' in dataset and method in ['wnngip', 'gip']:
                ## reduce size of train set
                # pass
                  print(drugMat.shape, targetMat.shape)
                  ii = np.argsort(drugMat[i, :])[::-1][:1000]
                  jj = np.argsort(targetMat[j, :])[::-1][:1000]
                  W_, intMat_, drugMat_, targetMat_ = \
                      W[ii, :], intMat[ii, :], drugMat[ii, :], targetMat[jj, :]
                  W_, intMat_, drugMat_, targetMat_ = \
                      W_[:, jj], intMat_[:, jj], drugMat_[:, ii], targetMat_[:, jj]
                  test_data_ = np.asarray([[0, 0]])
                  R = W_ * intMat_
                  model.fix_model(W_, intMat_, drugMat_, targetMat_, seed)
                  model.predict(test_data_, R, test_data)
              else:
                  R = W * intMat
                  model.fix_model(W, intMat, drugMat, targetMat, seed)
                  model.predict(test_data, R)
              print(time.time() - t1)
          iii += 1
        pred_ind = np.where(model.pred != np.inf)
        pred_local = model.pred[pred_ind[0], pred_ind[1]]
        test_local = intMat[pred_ind[0], pred_ind[1]]
        aupr_val, auc_val = model.get_perf(intMat)
        print('aupr_val, auc_val', aupr_val, auc_val)
        return aupr_val, auc_val, pred_local, test_local
    else:
        print("wrong cv_type")
        exit(1)


def train(model, method, dataset, cv_data, intMat, drugMat, targetMat, cv_type='fold'):
    if cv_type in ['fold', 'fold_balanced', 'cluster_fold_balanced']:
        aupr, auc = [], []
        for seed in cv_data.keys():
            for W, test_data, test_label in cv_data[seed]:
                # tic, toc = time.time(), time.clock()
                model.pred = np.full(intMat.shape, np.inf)
                model.fix_model(W, intMat, drugMat, targetMat, seed)
                R = W * intMat
                aupr_val, auc_val, _ = model.evaluation(test_data, test_label, intMat, R)
                # print("time", time.time() - tic, time.clock() - toc)
                aupr.append(aupr_val)
                auc.append(auc_val)
    elif cv_type == 'loo':
        seed = list(cv_data.keys())[0]
        model.pred = np.full(intMat.shape, np.inf)
        for i in range(intMat.shape[0]):
            for j in range(intMat.shape[1]):
                W = np.ones(intMat.shape)
                W[i, j], test_data, test_label = 0, np.asarray([[i, j]]), intMat[i, j]
                R = W * intMat
                model.fix_model(W, intMat, drugMat, targetMat, seed)
                model.predict(test_data, R)
        aupr, auc = model.get_perf(intMat)
    elif cv_type == 'loo_balanced':
        aupr, auc = [], []
        for seed in cv_data.keys():
            model.pred = np.full(intMat.shape, np.inf)
            for i, j in cv_data[seed]:
                W = np.ones(intMat.shape)
                W[i, j], test_data, test_label = 0, np.asarray([[i, j]]), intMat[i, j]
                model.fix_model(W, intMat, drugMat, targetMat, seed)
                R = W * intMat
                model.predict(test_data, R)
            aupr_val, auc_val = model.get_perf(intMat)
            aupr.append(aupr_val)
            auc.append(auc_val)
    elif cv_type == 'loo_balanced_no_orphans':
        aupr, auc = [], []
        for seed in cv_data.keys():
            model.pred = np.full(intMat.shape, np.inf)
            for i, j in cv_data[seed]:
              W = np.ones(intMat.shape)
              W[i, j], test_data, test_label = 0, np.asarray([[i, j]]), intMat[i, j]
              if len(np.where(intMat[i, :] == 1)[0]) != 0 and \
                      len(np.where(intMat[:, j] == 1)[0]) != 0:
                  R = W * intMat
                  model.fix_model(W, intMat, drugMat, targetMat, seed)
                  model.predict(test_data, R)
            aupr_val, auc_val = model.get_perf(intMat)
            aupr.append(aupr_val)
            auc.append(auc_val)
    elif cv_type == 'hard_loo_balanced':
        aupr, auc = [], []
        for seed in cv_data.keys():
            model.pred = np.full(intMat.shape, np.inf)
            print('n couple', len(cv_data[seed]))
            ind = 0
            for i, j in cv_data[seed]:
                t1 = time.time()
                W = np.ones(intMat.shape)
                W[i, j], test_data, test_label = 0, np.asarray([[i, j]]), intMat[i, j]
                # W[i, np.where(intMat[i, :] == 1)[0]] = 0
                # W[np.where(intMat[:, j] == 1)[0], j] = 0
                W[i, :].fill(0)
                W[:, j].fill(0)
                if 'drugbank' in dataset and method in ['wnngip', 'nlrmf']:
                    ## reduce size of train set
                    # pass
                    ii = np.argsort(drugMat[i, :])[::-1][:200]
                    jj = np.argsort(targetMat[j, :])[::-1][:200]
                    W_, intMat_, drugMat_, targetMat_ = \
                        W[ii, :], intMat[ii, :], drugMat[ii, :], targetMat[jj, :]
                    W_, intMat_, drugMat_, targetMat_ = \
                        W_[:, jj], intMat_[:, jj], drugMat_[:, ii], targetMat_[:, jj]
                    test_data_ = np.asarray([[0, 0]])
                    R = W_ * intMat_
                    model.fix_model(W_, intMat_, drugMat_, targetMat_, seed)
                    model.predict(test_data_, R, test_data)
                else:
                    R = W * intMat
                    model.fix_model(W, intMat, drugMat, targetMat, seed)
                    model.predict(test_data, R)
                # print(ind, time.time() - t1)
                ind += 1
            aupr_val, auc_val = model.get_perf(intMat)
            aupr.append(aupr_val)
            auc.append(auc_val)
    else:
        print("wrong cv_type")
        exit(1)
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
