import time
from functions import *
from nrlmf import NRLMF
from netlaprls import NetLapRLS
from blm import BLMNII
from wnngip import WNNGIP, GIP, NNWNNGIP, NNGIP
# from kbmf import KBMF
from cmf import CMF
from nnkronsvm import NNKronSVM, NNKronSVMGIP, NNKronWNNSVMGIP, NNKronWNNSVM
import pickle


def get_list_param_and_dict_perf(method):
    if method in ['wnngip', 'gip']:
        dict_perf = {}
        list_param = []
        for x in np.arange(0.1, 1.1, 0.1):
            dict_perf[x] = {}
            for y in np.arange(0.0, 1.1, 0.1):
                list_param.append((x, y))
    elif method in ['nnwnngip', 'nngip']:
        dict_perf = {}
        list_param = []
        for x in np.arange(0.1, 1.1, 0.1):
            dict_perf[x] = {}
            for y in np.arange(0.0, 1.1, 0.1):
                dict_perf[x][y] = {}
                for NN in [1, 2, 3, 5, 10, 20]:
                    list_param.append((x, y, NN))
    elif method == 'nrlmf':
        list_param = []
        dict_perf = {}
        for r in [50, 100]:
            dict_perf[r] = {}
            for x in np.arange(-5, 2):
                dict_perf[r][x] = {}
                for y in np.arange(-5, 3):
                    dict_perf[r][x][y] = {}
                    for z in np.arange(-5, 1):
                        dict_perf[r][x][y][z] = {}
                        for t in np.arange(-3, 1):
                            list_param.append((r, x, y, z, t))
    elif method in ['nnkronsvm', 'nnkronsvmgip']:
        list_param = []
        dict_perf = {}
        for C in [.01, .05, .1, .5, 1., 10., 100., ]:
          # dict_perf[C] = {}
          for posnei in [2, 5, 10]:
            # dict_perf[C][posnei] = {}
            for negnei in [1, 2, 5]:
                list_param.append((C, posnei, negnei))
    elif method in ['nnkronwnnsvmgip', 'nnkronwnnsvm']:
        list_param = []
        dict_perf = {}
        for C in [.01, .05, .1, .5, 1., 10., 100., ]:
          dict_perf[C] = {}
          for posnei in [2, 5, 10]:
            dict_perf[C][posnei] = {}
            for negnei in [1, 2, 5]:
              dict_perf[C][posnei][negnei] = {}
              for t in np.arange(0.1, 1.1, 0.1):
                  list_param.append((C, posnei, negnei, t))
    return list_param, dict_perf


def get_model(method, para, par, dataset):
    if method == 'wnngip':
      model = WNNGIP(T=par[0], sigma=1, alpha=par[1])
    elif method == 'gip':
        model = GIP(T=par[0], sigma=1, alpha=par[1])
    elif method == 'nngip':
      model = NNGIP(T=par[0], sigma=1, alpha=par[1], NN=par[2])
    elif method == 'nnwnngip':
      model = NNWNNGIP(T=par[0], sigma=1, alpha=par[1], NN=par[2])
    elif method == 'nrlmf':
      model = NRLMF(cfix=para['c'], K1=para['K1'], K2=para['K2'], num_factors=par[0],
                    lambda_d=2**(par[1]), lambda_t=2**(par[1]), alpha=2**(par[2]),
                    beta=2**(par[3]), theta=2**(par[4]), max_iter=100)
    elif method == 'nnkronsvm':
      model = NNKronSVM(C=par[0], NbNeg=para['NbNeg'], NegNei=par[2],
                        PosNei=par[1], dataset=dataset, n_proc=1)
    elif method == 'nnkronsvmgip':
      model = NNKronSVMGIP(C=par[0], NbNeg=para['NbNeg'], NegNei=par[2],
                           PosNei=par[1], dataset=dataset, n_proc=1)
    elif method == 'nnkronwnnsvmgip':
      model = NNKronWNNSVMGIP(C=par[0], t=par[3], NbNeg=para['NbNeg'], NegNei=par[2],
                              PosNei=par[1], dataset=dataset, n_proc=1)
    elif method == 'nnkronwnnsvm':
      model = NNKronWNNSVM(C=par[0], t=par[3], NbNeg=para['NbNeg'], NegNei=par[2],
                           PosNei=par[1], dataset=dataset, n_proc=1)
    return model


def feed_dict_perf(method, para, par, dict_perf, aupr_vec, auc_vec, pred, test, tic, toc):
    if method in ['wnngip', 'gip']:
      dict_perf[par[0]][par[1]] = (aupr_vec, auc_vec, pred, test, tic, toc)
    elif method in ['nngip', 'nnwnngip']:
      dict_perf[par[0]][par[1]][par[2]] = (aupr_vec, auc_vec, pred, test, tic, toc)
    elif method == 'nrlmf':
      dict_perf[par[0]][par[1]][par[2]][par[3]][par[4]] = (aupr_vec, auc_vec, pred, test, tic, toc)
    elif method in ['nnkronsvm', 'nnkronsvmgip']:
      dict_perf[par[0]] = (aupr_vec, auc_vec, pred, test, tic, toc)  # bug here
    elif method in ['nnkronwnnsvmgip', 'nnkronwnnsvm']:
      dict_perf[par[0]][par[1]][par[2]][par[3]] = (aupr_vec, auc_vec, pred, test, tic, toc)
    return dict_perf


def eval(method, dataset, cv_data, X, D, T, cvs, para, cv_type, i_param, i_test):
    list_param, dict_perf = get_list_param_and_dict_perf(method)

    par = list_param[i_param]

    tic, toc = time.clock(), time.time()

    print('get model')
    model = get_model(method, para, par, dataset)

    cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + " CV: " + cv_type +\
        "\n" + str(model)
    print(cmd)

    aupr_vec, auc_vec, pred, test = train_single(model, method, dataset, cv_data,
                                                 X, D, T, cv_type, i_test)

    dict_perf = feed_dict_perf(method, para, par, dict_perf, aupr_vec, auc_vec, pred, test,
                               time.clock() - tic, time.time() - toc)

    m = get_name_method(method)

    data_file = dataset + "_" + str(cvs) + "_" + cv_type + '_Model:' + m + '_' + \
        str(i_param) + '_' + str(i_test)
    pickle.dump(dict_perf, open('results/' + data_file + '.data', 'wb'))


def cv_eval(method, dataset, cv_data, X, D, T, cvs, para, cv_type):
    list_param, dict_perf = get_list_param_and_dict_perf(method)

    max_auc, auc_opt = 0, []
    for par in list_param:
        tic, toc = time.clock(), time.time()

        print('get_model')
        model = get_model(method, para, par, dataset)

        cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + " CV: " + cv_type +\
            "\n" + str(model)
        print(cmd)

        aupr_vec, auc_vec = train(model, method, dataset, cv_data, X, D, T, cv_type)

        if cv_type != 'loo':
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        else:
            aupr_avg, aupr_conf, auc_avg, auc_conf = aupr_vec, 0., auc_vec, 0.
        print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f,%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic, time.time() - toc))
        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
        dict_perf = feed_dict_perf(method, para, par, dict_perf, aupr_vec, auc_vec,
                                   aupr_avg, auc_avg,
                                   time.clock() - tic, time.time() - toc)

    m = get_name_method(method)
    data_file = dataset + "_" + str(cvs) + "_" + cv_type + '_Model:' + m
    pickle.dump(dict_perf, open('results/' + data_file + '.data', 'wb'))

    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


