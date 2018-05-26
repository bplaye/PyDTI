#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from sklearn import svm
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import copy
import pickle
import pandas as pd
import csv
import multiprocessing
from sklearn.metrics.pairwise import rbf_kernel


def get_pred(tup):
  model, ind_couples, couple_test = tup
  return model.get_pred(ind_couples, couple_test)


def get_pred_joblib(tup):
  model, ind_couples, couple_test = tup
  return model.get_pred(ind_couples, couple_test), ind_couples


def get_pred_joblib2(list_tup):
  list_pred = []
  for tup in list_tup:
    model, ind_couples, couple_test = tup
    pred = model.get_pred(ind_couples, couple_test)
    list_pred.append((pred, ind_couples))
  return list_pred


def randomize_list(list_to_be_randomized):
    list_to_be_randomized = copy.deepcopy(list_to_be_randomized)
    randomized_list = []
    while len(list_to_be_randomized) > 0:
        rand_int = np.random.randint(len(list_to_be_randomized))
        randomized_list.append(list_to_be_randomized[rand_int])
        del list_to_be_randomized[rand_int]
    return randomized_list


class NNKronSVM():
  def __init__(self, C, NbNeg, NegNei, PosNei, dataset, n_proc):
    self.NbNeg = NbNeg
    self.PosNei = PosNei
    self.NegNei = NegNei
    self.C = C
    self.n_proc = n_proc
    self.alpha = 0.5
    self.gamma = 1.0

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
    self.dico_neerest_nei_per_prot_ID = \
        pickle.load(open('data/' + dataset + '_dico_neerest_nei_per_prot_ID.data', 'rb'))
    print(1)
    self.dico_neerest_nei_per_prot_value = \
        pickle.load(open('data/' + dataset + '_dico_neerest_nei_per_prot_value.data', 'rb'))
    print(2)
    self.dico_neerest_nei_per_mol_ID = \
        pickle.load(open('data/' + dataset + '_dico_neerest_nei_per_mol_ID.data', 'rb'))
    print(3)
    self.dico_neerest_nei_per_mol_value = \
        pickle.load(open('data/' + dataset + '_dico_neerest_nei_per_mol_value.data', 'rb'))
    print(4)

    self.dico_target_of_mol = \
        pickle.load(open('data/' + dataset + '_dico_target_of_mol.data', 'rb'))
    print(1)
    self.dico_ligand_of_prot = \
        pickle.load(open('data/' + dataset + '_dico_ligand_of_prot.data', 'rb'))
    print(2)
    self.dico_labels_per_couple = \
        pickle.load(open('data/' + dataset + '_dico_labels_per_couple.data', 'rb'))
    print(3)

    self.dico_prot2indice = pickle.load(open('data/' + dataset + '_dico_prot2indice.data', 'rb'))
    print(1)
    self.dico_mol2indice = pickle.load(open('data/' + dataset + '_dico_mol2indice.data', 'rb'))
    print(2)
    self.dico_indice2prot = pickle.load(open('data/' + dataset + '_dico_indice2prot.data', 'rb'))
    print(3)
    self.dico_indice2mol = pickle.load(open('data/' + dataset + '_dico_indice2mol.data', 'rb'))
    print(4)

    self.threshold_prot = None
    self.threshold_mol = None

  def get_pred(self, couple_test, R):
    list_train_samples = []
    list_train_labels = []

    list_train_samples_mol, list_train_labels_mol = self.update_with_intra_task_pairs(
        self.dico_target_of_mol[couple_test[1]][0], self.dico_target_of_mol[couple_test[1]][1],
        [couple_test[0]], self.K_prot, self.dico_prot2indice, self.threshold_prot)
    list_train_samples_prot, list_train_labels_prot = self.update_with_intra_task_pairs(
        self.dico_ligand_of_prot[couple_test[0]][0], self.dico_ligand_of_prot[couple_test[0]][1],
        [couple_test[1]], self.K_mol, self.dico_mol2indice, self.threshold_mol)

    for _, el in enumerate(list_train_samples_mol):
      if (el, couple_test[1]) not in self.forbidden_samples:
        list_train_samples.append((el, couple_test[1]))
        list_train_labels.append(list_train_labels_mol[_])
    for _, el in enumerate(list_train_samples_prot):
      if (couple_test[0], el) not in self.forbidden_samples:
        list_train_samples.append((couple_test[0], el))
        list_train_labels.append(list_train_labels_prot[_])

    list_train_samples, list_train_labels = self.update_with_extra_task_pairs(
        couple_test, list_train_samples, list_train_labels)

    K_train, K_test = self.make_Ktrain_and_Ktest_MT(list_train_samples, [couple_test])

    clf = svm.SVC(kernel='precomputed', C=self.C, class_weight='balanced')
    clf.fit(K_train, list_train_labels)
    return (clf.decision_function(K_test).tolist())[0]

  def kernel_combination(self, R, S, new_inx, bandwidth):
    return S

  def get_perf(self, intMat):
    pred_ind = np.where(self.pred != np.inf)
    pred_local = self.pred[pred_ind[0], pred_ind[1]]
    test_local = intMat[pred_ind[0], pred_ind[1]]
    prec, rec, thr = precision_recall_curve(test_local, pred_local)
    aupr_val = auc(rec, prec)
    fpr, tpr, thr = roc_curve(test_local, pred_local)
    auc_val = auc(fpr, tpr)
    return aupr_val, auc_val

  def fix_model(self, W, intMat, drugMat, targetMat, seed, epsilon=0.1):

    R = W * intMat
    m, n = intMat.shape
    x, y = np.where(R > 0)
    # # Enforce the positive definite property of similarity matrix
    drugMat = (drugMat + drugMat.T) / 2 + epsilon * np.eye(m)
    targetMat = (targetMat + targetMat.T) / 2 + epsilon * np.eye(n)
    # train_drugs = np.array(list(set(x.tolist())), dtype=np.int32)
    # train_targets = np.array(list(set(y.tolist())), dtype=np.int32)
    new_drugs = np.array(list(set(xrange(m)) - set(x.tolist())), dtype=np.int32)
    new_targets = np.array(list(set(xrange(n)) - set(y.tolist())), dtype=np.int32)
    drug_bw = self.gamma * m / len(x)
    target_bw = self.gamma * n / len(x)
    self.K_mol = self.kernel_combination(R, drugMat, new_drugs, drug_bw)
    self.K_prot = self.kernel_combination(R.T, targetMat, new_targets, target_bw)

    forbidden_data = np.where(W == 0)
    self.forbidden_samples = [(self.dico_indice2prot[forbidden_data[1][i]],
                               self.dico_indice2mol[forbidden_data[0][i]])
                              for i in range(len(forbidden_data[0]))]
    # print('forbidden', len(self.forbidden_samples), self.forbidden_samples)
    # self.list_ind_test = [(test_data[1][i], test_data[0][i]) for i in range(len(test_data[0]))]

  def evaluation(self, test_data, test_label, intMat, R):
    self.predict(test_data, R)
    aupr_val, auc_val = self.get_perf(intMat)
    pred_ind = np.where(self.pred != np.inf)
    pred_local = self.pred[pred_ind[0], pred_ind[1]]
    return aupr_val, auc_val, pred_local

  def predict(self, test_data, R):
    list_couple_test = [(self.dico_indice2prot[test_data[i][1]],
                         self.dico_indice2mol[test_data[i][0]])
                        for i in range(test_data.shape[0])]
    list_ind_test = [(test_data[i][1], test_data[i][0]) for i in range(test_data.shape[0])]
    for ind_couples in range(len(list_couple_test)):
        couple_test = list_couple_test[ind_couples]
        pred = self.get_pred(couple_test, R)
        self.pred[list_ind_test[ind_couples][1], list_ind_test[ind_couples][0]] = pred


    # if self.n_proc == 1:
    #   for ind_couples in range(len(self.list_couple_test)):
    #     couple_test = self.list_couple_test[ind_couples]
    #     pred = self.get_pred(ind_couples, couple_test)
    #     self.pred[self.list_ind_test[ind_couples][1], self.list_ind_test[ind_couples][0]] = pred

    # else:
    #   if False:
    #     from multiprocessing.pool import Pool
    #     pool = Pool(processes=self.n_proc)
    #     async_result = []

    #     # for ind_couples in range(len(self.list_couple_test)):
    #     #   couple_test = self.list_couple_test[ind_couples]
    #     #   async_result.append(pool.apply_async(get_pred, (self, ind_couples, couple_test)))
    #     # multiple_results = [res.get() for res in async_result]

    #     async_result = pool.map_async(get_pred, [(self, i_, c_)
    #                                              for i_, c_ in enumerate(self.list_couple_test)])
    #     multiple_results = async_result.get()

    #     pool.close()
    #     pool.join()

    #     # print(multiple_results)
    #     for ind_couples, pred in enumerate(multiple_results):
    #       self.pred[self.list_ind_test[ind_couples][1], self.list_ind_test[ind_couples][0]] = pred
    #   elif True:
    #     from multiprocessing.pool import Pool
    #     pool = Pool(processes=self.n_proc)
    #     async_result = []

    #     n_ = int(np.floor(float(len(self.list_couple_test)) / float(self.n_proc)))
    #     l_ = []
    #     for _ in range(self.n_proc):
    #       if _ < self.n_proc - 1:
    #         l_.append([(self, i, self.list_couple_test[i]) for i in range(_ * n_, (_ + 1) * n_)])
    #       else:
    #         l_.append([(self, i, self.list_couple_test[i]) for i in range(_ * n_, len(self.list_couple_test))])

    #     async_result = pool.map_async(get_pred_joblib2, l_)
    #     multiple_results = async_result.get()

    #     pool.close()
    #     pool.join()

    #     for list_res in multiple_results:
    #       for pred, i_ in list_res:
    #         self.pred[self.list_ind_test[i_][1], self.list_ind_test[i_][0]] = pred

  def __str__(self,):
    return "Model:NNKronSVM, NbNeg:%s, NegNei:%s, PosNei:%s, C:%s" % (self.NbNeg, self.NegNei,
                                                                      self.PosNei, self.C)

  def update_with_intra_task_pairs(self, list_pos, list_neg_samples, list_test_samples, Kernel,
                                   dico_2indice, threshold, value_of_neg_class=0):
    list_train_labels = []
    list_train_samples = []

    for pos_train_sample in list_pos:
        if pos_train_sample not in list_test_samples:
            if self.condition_on_intra_task(Kernel, dico_2indice, pos_train_sample,
                                            list_test_samples[0], threshold):
                list_train_samples.append(pos_train_sample)
                list_train_labels.append(1)
    # print(list_test_samples)
    # print('len(list_train_samples)', len(list_train_samples), list_train_samples)

    if self.NbNeg != "full":
        nb_pos_sample = len(list_train_samples) if len(list_train_samples) > 0 else 2
        nb_neg_sample = 0
        rand_list_of_indices = randomize_list([i for i in range(len(list_neg_samples))])
        for rand_int in rand_list_of_indices:
            if list_neg_samples[rand_int] not in list_pos and \
                    list_neg_samples[rand_int] not in list_test_samples and \
                    list_neg_samples[rand_int] not in list_train_samples and \
                    nb_neg_sample < nb_pos_sample * self.NbNeg:
                list_train_samples.append(list_neg_samples[rand_int])
                list_train_labels.append(value_of_neg_class)
                nb_neg_sample += 1
    else:
        for neg_sample in list_neg_samples:
            if neg_sample not in list_test_samples and neg_sample not in list_pos and neg_sample not in list_train_samples:
                list_train_samples.append(neg_sample)
                list_train_labels.append(value_of_neg_class)
    # print(list_train_samples)
    # print('len(list_train_samples)', len(list_train_samples))
    return list_train_samples, list_train_labels

  def update_with_extra_task_pairs(self, current_couple, list_train_samples, list_train_labels, value_of_neg_class=0):
        """
        adding extra-task instances to the train set chosen as closer as possible to the tested sample
        """
        # print('current couple:', current_couple)
        sys.stdout.flush()
        dico_neerest_pos_nei_per_couple_ID = []
        dico_neerest_neg_nei_per_couple_ID = []

        array_of_sim_ind = np.zeros(len(self.dico_neerest_nei_per_prot_ID[current_couple[0]]), dtype=np.int)
        ## chaque prot de dico_neerest_nei_per_prot_ID est associee à un indice qui le lie à la molécule avec qui il formera le plus couple de plus forte similarite (qu'il n'a pas encore fait -d'ou le fait de commencer à l'indice 0 du dico_neerest_nei_per_mol_value)
        array_of_sim_value = np.zeros(len(array_of_sim_ind))
        ## pour chaque prot de dico_neerest_nei_per_prot_ID, on calcul la valeure de similarité de couple la plus forte que chaque prot puisse faire
        for ind in range(len(array_of_sim_ind)):
            array_of_sim_value[ind] = self.dico_neerest_nei_per_prot_value[current_couple[0]][ind] * self.dico_neerest_nei_per_mol_value[current_couple[1]][array_of_sim_ind[ind]]
        sorted_line = np.argsort(array_of_sim_value)

        nb_pos_local = self.PosNei
        nb_neg_local = self.PosNei * self.NegNei
        pos_local = 0
        neg_local = 0

        while pos_local < nb_pos_local or neg_local < nb_neg_local:
            max_ind = np.argmax(array_of_sim_value)
            compared_couple = (self.dico_neerest_nei_per_prot_ID[current_couple[0]][max_ind], self.dico_neerest_nei_per_mol_ID[current_couple[1]][array_of_sim_ind[max_ind]])
#            print(self.dico_labels_per_couple[compared_couple[1] + compared_couple[0]])
#            sys.stdout.flush()
            if compared_couple not in list_train_samples:
              if compared_couple not in self.forbidden_samples:
                if compared_couple[0] != current_couple[0] and compared_couple[1] != current_couple[1]:
                    if self.condition_on_extra_task(current_couple, compared_couple):
                        # print(compared_couple)
                        if self.dico_labels_per_couple[compared_couple[1] + compared_couple[0]] == 1 and pos_local < nb_pos_local:
                            dico_neerest_pos_nei_per_couple_ID.append(compared_couple)
                            pos_local += 1
                        elif self.dico_labels_per_couple[compared_couple[1] + compared_couple[0]] == value_of_neg_class and neg_local < nb_neg_local:
                            dico_neerest_neg_nei_per_couple_ID.append(compared_couple)
                            neg_local += 1

            array_of_sim_ind[max_ind] += 1
            if array_of_sim_ind[max_ind] == len(self.dico_neerest_nei_per_mol_ID[current_couple[1]]):
                array_of_sim_value[max_ind] = -1
            else:
                array_of_sim_value[max_ind] = self.dico_neerest_nei_per_prot_value[current_couple[0]][max_ind] * self.dico_neerest_nei_per_mol_value[current_couple[1]][array_of_sim_ind[max_ind]]
            if all(i >= len(self.dico_neerest_nei_per_mol_ID[current_couple[1]]) for i in array_of_sim_ind):
                pos_local = nb_pos_local
                neg_local = nb_neg_local
                print('WARNING : all couples were spanned when finding neighboors')
                sys.stdout.flush()

        if current_couple in dico_neerest_pos_nei_per_couple_ID:
                print("current couple in list de neerest nei")
                sys.stdout.flush()
                exit(1)
        elif current_couple in dico_neerest_neg_nei_per_couple_ID:
                print("current couple in list de neerest nei")
                sys.stdout.flush()
                exit(1)

        list_train_samples_local = list_train_samples + dico_neerest_pos_nei_per_couple_ID + dico_neerest_neg_nei_per_couple_ID
        list_train_labels_local = list_train_labels + [1 for _ in range(len(dico_neerest_pos_nei_per_couple_ID))] + [value_of_neg_class for _ in range(len(dico_neerest_neg_nei_per_couple_ID))]

        return list_train_samples_local, list_train_labels_local

  def condition_on_extra_task(self, current_couple, compared_couple):
    return True

  def condition_on_intra_task(self, Kernel, dico_2indice, compared_sample, tested_sample,
                              threshold):
    return True

  def make_Ktrain_and_Ktest_MT(self, list_train_samples, list_test_samples):
    """
    computes the kernels associated respectively to the training and testing sets when working on the chemogenomic space.
    (i.e. the kronecker over the molecular and protein spaces)
    """
    K_train = np.zeros((len(list_train_samples), len(list_train_samples)))
    K_test = np.zeros((len(list_test_samples), len(list_train_samples)))
    for ind1 in range(len(list_train_samples)):
        if list_train_samples[ind1] in list_test_samples:
            raise ValueError("exit because train is in test")
        ind1_Kprot = self.dico_prot2indice[list_train_samples[ind1][0]]
        ind1_Kmol = self.dico_mol2indice[list_train_samples[ind1][1]]
        for ind2 in range(ind1, len(list_train_samples)):
            ind2_Kprot = self.dico_prot2indice[list_train_samples[ind2][0]]
            ind2_Kmol = self.dico_mol2indice[list_train_samples[ind2][1]]
            K_train[ind1, ind2] = self.K_prot[ind1_Kprot, ind2_Kprot] * self.K_mol[ind1_Kmol, ind2_Kmol]
            K_train[ind2, ind1] = K_train[ind1, ind2]
        for ind_t in range(len(list_test_samples)):
            ind_t_Kprot = self.dico_prot2indice[list_test_samples[ind_t][0]]
            ind_t_Kmol = self.dico_mol2indice[list_test_samples[ind_t][1]]
            K_test[ind_t, ind1] = \
                self.K_prot[ind1_Kprot, ind_t_Kprot] * self.K_mol[ind1_Kmol, ind_t_Kmol]

    return K_train, K_test


class NNKronSVMGIP(NNKronSVM):

  def kernel_combination(self, R, S, new_inx, bandwidth):
    K = self.alpha * S + (1.0 - self.alpha) * rbf_kernel(R, gamma=bandwidth)
    K[new_inx, :] = S[new_inx, :]
    K[:, new_inx] = S[:, new_inx]
    return K


class NNKronWNNSVMGIP(NNKronSVMGIP):

  def __init__(self, C, t, NbNeg, NegNei, PosNei, dataset, n_proc):
    self.NbNeg = NbNeg
    self.PosNei = PosNei
    self.NegNei = NegNei
    self.C = C
    self.n_proc = n_proc
    self.alpha = 0.5
    self.gamma = 1.0
    self.t = t

    if 'drugbank' in dataset:
        if '2000' in dataset:
            limit = 2000
        elif '5000' in dataset:
            limit = 5000
        else:
            limit = None
        dataset = 'DrugBank'

    self.dico_neerest_nei_per_prot_ID = \
        pickle.load(open('data/' + dataset + '_dico_neerest_nei_per_prot_ID.data', 'rb'))
    self.dico_neerest_nei_per_prot_value = \
        pickle.load(open('data/' + dataset + '_dico_neerest_nei_per_prot_value.data', 'rb'))
    self.dico_neerest_nei_per_mol_ID = \
        pickle.load(open('data/' + dataset + '_dico_neerest_nei_per_mol_ID.data', 'rb'))
    self.dico_neerest_nei_per_mol_value = \
        pickle.load(open('data/' + dataset + '_dico_neerest_nei_per_mol_value.data', 'rb'))

    self.dico_target_of_mol = \
        pickle.load(open('data/' + dataset + '_dico_target_of_mol.data', 'rb'))
    self.dico_ligand_of_prot = \
        pickle.load(open('data/' + dataset + '_dico_ligand_of_prot.data', 'rb'))
    self.dico_labels_per_couple = \
        pickle.load(open('data/' + dataset + '_dico_labels_per_couple.data', 'rb'))

    self.dico_prot2indice = pickle.load(open('data/' + dataset + '_dico_prot2indice.data', 'rb'))
    self.dico_mol2indice = pickle.load(open('data/' + dataset + '_dico_mol2indice.data', 'rb'))
    self.dico_indice2prot = pickle.load(open('data/' + dataset + '_dico_indice2prot.data', 'rb'))
    self.dico_indice2mol = pickle.load(open('data/' + dataset + '_dico_indice2mol.data', 'rb'))

    self.threshold_prot = None
    self.threshold_mol = None

  def get_pred(self, couple_test, R):
    # print(couple_test)
    # mol side
    list_train_mol_samples, list_train_mol_labels = [], []
    list_train_samples_mol, list_train_labels_mol = self.update_with_intra_task_pairs(
        self.dico_target_of_mol[couple_test[1]][0], self.dico_target_of_mol[couple_test[1]][1],
        [couple_test[0]], self.K_prot, self.dico_prot2indice, self.threshold_prot)
    for _, el in enumerate(list_train_samples_mol):
      if (el, couple_test[1]) not in self.forbidden_samples:
        list_train_mol_samples.append((el, couple_test[1]))
        list_train_mol_labels.append(list_train_labels_mol[_])
    if 1 not in list_train_mol_labels:  # orphan mol, we consider WNN
      # print('orphan mol')
      inx = np.argsort(self.K_mol[self.dico_mol2indice[couple_test[1]], :])[::-1]
      nei_local, list_pos_local, list_neg_local = np.zeros(self.K_prot.shape[0]), [], []
      i = 0
      for _ in xrange(len(inx)):
        if np.any(R[inx[_], :] == 1) == True:
          w = self.t**(i)
          # print('w', w)
          if w >= 1e-4:
            nei_local += w * R[inx[_], :]
          else:
            break
          i += 1
      for i in range(len(nei_local)):
        if nei_local[i] >= 0.5:
            list_pos_local.append(self.dico_indice2prot[i])
        else:
            list_neg_local.append(self.dico_indice2prot[i])
      list_train_mol_samples, list_train_mol_labels = [], []
      list_train_samples_mol, list_train_labels_mol = self.update_with_intra_task_pairs(
          list_pos_local, list_neg_local,
          [couple_test[0]], self.K_prot, self.dico_prot2indice, self.threshold_prot)
      for _, el in enumerate(list_train_samples_mol):
          list_train_mol_samples.append((el, couple_test[1]))
          list_train_mol_labels.append(list_train_labels_mol[_])
      # print('size mol train', len(np.where(np.asarray(list_train_mol_labels) == 1)[0]))

    # prot side
    list_train_prot_samples, list_train_prot_labels = [], []
    list_train_samples_prot, list_train_labels_prot = self.update_with_intra_task_pairs(
        self.dico_ligand_of_prot[couple_test[0]][0], self.dico_ligand_of_prot[couple_test[0]][1],
        [couple_test[1]], self.K_mol, self.dico_mol2indice, self.threshold_mol)
    for _, el in enumerate(list_train_samples_prot):
      if (couple_test[0], el) not in self.forbidden_samples:
        list_train_prot_samples.append((couple_test[0], el))
        list_train_prot_labels.append(list_train_labels_prot[_])
    if 1 not in list_train_prot_labels:  # orphan mol, we consider WNN
      # print('orphan prot')
      inx = np.argsort(self.K_prot[self.dico_prot2indice[couple_test[0]], :])[::-1]
      nei_local, list_pos_local, list_neg_local = np.zeros(self.K_mol.shape[0]), [], []
      i = 0
      for _ in xrange(len(inx)):
        if np.any(R[:, inx[_]] == 1) == True:
          w = self.t**(i)
          # print('w', w)
          if w >= 1e-4:
            nei_local += w * R[:, inx[_]]
          else:
            break
          i += 1
      for i in range(len(nei_local)):
        if nei_local[i] >= 0.5:
            list_pos_local.append(self.dico_indice2mol[i])
        else:
            list_neg_local.append(self.dico_indice2mol[i])
      list_train_prot_samples, list_train_prot_labels = [], []
      list_train_samples_prot, list_train_labels_prot = self.update_with_intra_task_pairs(
          list_pos_local, list_neg_local,
          [couple_test[1]], self.K_mol, self.dico_mol2indice, self.threshold_mol)
      for _, el in enumerate(list_train_samples_prot):
          list_train_prot_samples.append((couple_test[0], el))
          list_train_prot_labels.append(list_train_labels_prot[_])
      # print('size prot train', len(np.where(np.asarray(list_train_prot_labels) == 1)[0]))

    list_train_samples = list_train_mol_samples + list_train_prot_samples
    list_train_labels = list_train_mol_labels + list_train_prot_labels
    if couple_test in list_train_samples:
        print('train in test')
        exit(1)

    list_train_samples, list_train_labels = self.update_with_extra_task_pairs(
        couple_test, list_train_samples, list_train_labels)

    K_train, K_test = self.make_Ktrain_and_Ktest_MT(list_train_samples, [couple_test])

    clf = svm.SVC(kernel='precomputed', C=self.C, class_weight='balanced')
    clf.fit(K_train, list_train_labels)
    return (clf.decision_function(K_test).tolist())[0]

  def __str__(self,):
    return "Model:NNKronWNNSVMGIP, NbNeg:%s, NegNei:%s, PosNei:%s, C:%s, T:%s" % \
        (self.NbNeg, self.NegNei, self.PosNei, self.C, self.t)


class NNKronWNNSVM(NNKronWNNSVMGIP):

  def kernel_combination(self, R, S, new_inx, bandwidth):
    return S

  def __str__(self,):
    return "Model:NNKronWNNSVM, NbNeg:%s, NegNei:%s, PosNei:%s, C:%s, T:%s" % \
        (self.NbNeg, self.NegNei, self.PosNei, self.C, self.t)


if __name__ == "__main__":
  for dataset in ['e', 'gpcr', 'nr', 'ic']:
    print(dataset)
    dico_target_of_mol = {}
    dico_ligand_of_prot = {}

    dico_prot2indice = {}
    dico_mol2indice = {}
    dico_indice2prot = {}
    dico_indice2mol = {}

    dico_labels_per_couple = {}

    reader = csv.reader(open('data/' + dataset + '_admat_dgc.txt', 'r'), delimiter='\t')
    i = 0
    for row in reader:
      if i == 0:
        for j in range(1, len(row)):
          dico_mol2indice[row[j]], dico_indice2mol[j - 1] = j - 1, row[j]
          dico_target_of_mol[row[j]] = [[], []]
        list_mol = row[1:]
      else:
        dico_prot2indice[row[0]], dico_indice2prot[i - 1] = i - 1, row[0]
        dico_ligand_of_prot[row[0]] = [[], []]
        for j in range(1, len(row)):
          dico_labels_per_couple[list_mol[j - 1] + row[0]] = int(row[j])
          if int(row[j]) == 1:
            dico_ligand_of_prot[row[0]][0].append(list_mol[j - 1])
            dico_target_of_mol[list_mol[j - 1]][0].append(row[0])
          else:
            dico_ligand_of_prot[row[0]][1].append(list_mol[j - 1])
            dico_target_of_mol[list_mol[j - 1]][1].append(row[0])
      i += 1
    del reader
    print('len adamat:', i)

    print(dico_indice2prot.keys())

    pickle.dump(dico_target_of_mol, open('data/' + dataset + '_dico_target_of_mol.data', 'wb'))
    pickle.dump(dico_ligand_of_prot, open('data/' + dataset + '_dico_ligand_of_prot.data', 'wb'))
    pickle.dump(dico_prot2indice, open('data/' + dataset + '_dico_prot2indice.data', 'wb'))
    pickle.dump(dico_mol2indice, open('data/' + dataset + '_dico_mol2indice.data', 'wb'))
    pickle.dump(dico_indice2prot, open('data/' + dataset + '_dico_indice2prot.data', 'wb'))
    pickle.dump(dico_indice2mol, open('data/' + dataset + '_dico_indice2mol.data', 'wb'))
    pickle.dump(dico_labels_per_couple, open('data/' + dataset + '_dico_labels_per_couple.data',
                                             'wb'))

    for k in ['dc', 'dg']:
      Kernel = pd.read_csv('data/' + dataset + '_simmat_' + k + '.txt',
                           header=0, index_col=0, delimiter='\t').values
      if k == 'dc':
        dico_2indice = dico_mol2indice
        dico_indice2 = dico_indice2mol
      else:
        dico_2indice = dico_prot2indice
        dico_indice2 = dico_indice2prot

      dico_neerest_nei_per_ID = {}
      dico_neerest_nei_per_value = {}
      for cle, valeur in dico_2indice.items():
          dico_neerest_nei_per_ID[cle] = []
          dico_neerest_nei_per_value[cle] = []
          sorted_line = np.argsort(Kernel[valeur, :].copy())
          for ind in range(len(sorted_line)):
              indice = sorted_line[len(sorted_line) - 1 - ind]
              dico_neerest_nei_per_ID[cle].append(dico_indice2[indice])
              dico_neerest_nei_per_value[cle].append(Kernel[valeur, indice])
      if k == 'dc':
        pickle.dump(dico_neerest_nei_per_ID,
                    open('data/' + dataset + "_dico_neerest_nei_per_mol_ID.data", 'wb'))
        pickle.dump(dico_neerest_nei_per_value,
                    open('data/' + dataset + "_dico_neerest_nei_per_mol_value.data", 'wb'))
      else:
        pickle.dump(dico_neerest_nei_per_ID,
                    open('data/' + dataset + "_dico_neerest_nei_per_prot_ID.data", 'wb'))
        pickle.dump(dico_neerest_nei_per_value,
                    open('data/' + dataset + "_dico_neerest_nei_per_prot_value.data", 'wb'))

