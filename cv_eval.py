
import time
from functions import *
from nrlmf import NRLMF
from netlaprls import NetLapRLS
from blm import BLMNII
from wnngip import WNNGIP
# from kbmf import KBMF
from cmf import CMF
from nnkronsvm import NNKronSVM
import pickle


def nrlmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para, cv_type):
    dict_perf = {}
    max_auc, auc_opt = 0, []
    for r in [50, 100]:
        dict_perf[r] = {}
        for x in np.arange(-5, 2):
            dict_perf[r][x] = {}
            for y in np.arange(-5, 3):
                dict_perf[r][x][y] = {}
                for z in np.arange(-5, 1):
                    dict_perf[r][x][y][z] = {}
                    for t in np.arange(-3, 1):
                        tic = time.clock()
                        model = NRLMF(cfix=para['c'], K1=para['K1'], K2=para['K2'], num_factors=r, lambda_d=2**(x), lambda_t=2**(x), alpha=2**(y), beta=2**(z), theta=2**(t), max_iter=100)
                        cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + " CV: " + cv_type +\
                            "\n" + str(model)
                        print(cmd)
                        aupr_vec, auc_vec = train(model, cv_data, X, D, T, cv_type)
                        if cv_type != 'loo':
                            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                        else:
                            aupr_avg, aupr_conf, auc_avg, auc_conf = aupr_vec, 0., auc_vec, 0.
                        print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic))
                        if auc_avg > max_auc:
                            max_auc = auc_avg
                            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
                        dict_perf[r][x][y][z][t] = (aupr_vec, auc_vec)

    data_file = dataset + "_" + str(cvs) + "_" + cv_type + '_' + str(model)
    pickle.dump(dict_perf, open('results/' + data_file + '.data', 'wb'))

    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def netlaprls_cv_eval(method, dataset, cv_data, X, D, T, cvs, para, cv_type):
    dict_perf = {}
    max_auc, auc_opt = 0, []
    for x in np.arange(-6, 3):  # [-6, 2]
        dict_perf[x] = {}
        for y in np.arange(-6, 3):  # [-6, 2]
            tic = time.clock()
            model = NetLapRLS(gamma_d=10**(x), gamma_t=10**(x), beta_d=10**(y), beta_t=10**(y))
            cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + " CV: " + cv_type + "\n" +\
                str(model)
            print(cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T, cv_type)
            if cv_type != 'loo':
                aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            else:
                aupr_avg, aupr_conf, auc_avg, auc_conf = aupr_vec, 0., auc_vec, 0.
            print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic))
            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
            dict_perf[x][y] = (aupr_vec, auc_vec)

    data_file = dataset + "_" + str(cvs) + "_" + cv_type + '_' + str(model)
    pickle.dump(dict_perf, open('results/' + data_file + '.data', 'wb'))

    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def blmnii_cv_eval(method, dataset, cv_data, X, D, T, cvs, para, cv_type):
    dict_perf = {}
    max_auc, auc_opt = 0, []
    for x in np.arange(0, 1.1, 0.1):
        tic = time.clock()
        model = BLMNII(alpha=x, avg=False)
        cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + " CV: " + cv_type + "\n" +\
            str(model)
        print(cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T, cv_type)
        print(aupr_vec)
        if cv_type != 'loo':
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        else:
            aupr_avg, aupr_conf, auc_avg, auc_conf = aupr_vec, 0., auc_vec, 0.
        print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic))
        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
        dict_perf[x] = (aupr_vec, auc_vec)

    data_file = dataset + "_" + str(cvs) + "_" + cv_type + '_' + str(model)
    pickle.dump(dict_perf, open('results/' + data_file + '.data', 'wb'))

    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def wnngip_cv_eval(method, dataset, cv_data, X, D, T, cvs, para, cv_type):
    dict_perf = {}
    max_auc, auc_opt = 0, []
    for x in np.arange(0.1, 1.1, 0.1):
        dict_perf[x] = {}
        for y in np.arange(0.0, 1.1, 0.1):
            tic = time.clock()
            model = WNNGIP(T=x, sigma=1, alpha=y)
            cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + " CV: " + cv_type + "\n" +\
                str(model)
            print(cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T, cv_type)
            if cv_type != 'loo':
                aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            else:
                aupr_avg, aupr_conf, auc_avg, auc_conf = aupr_vec, 0., auc_vec, 0.
            print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic))
            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
            dict_perf[x][y] = (aupr_vec, auc_vec)

    data_file = dataset + "_" + str(cvs) + "_" + cv_type + '_' + str(model)
    pickle.dump(dict_perf, open('results/' + data_file + '.data', 'wb'))

    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def kbmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para, cv_type):
    dict_perf = {}
    max_auc, auc_opt = 0, []
    for d in [50, 100]:
        tic = time.clock()
        model = KBMF(num_factors=d)
        cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + " CV: " + cv_type + "\n" +\
            str(model)
        print(cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T, cv_type)
        if cv_type != 'loo':
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        else:
            aupr_avg, aupr_conf, auc_avg, auc_conf = aupr_vec, 0., auc_vec, 0.
        print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic))
        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
        dict_perf[d] = (aupr_vec, auc_vec)

    data_file = dataset + "_" + str(cvs) + "_" + cv_type + '_' + str(model)
    pickle.dump(dict_perf, open('results/' + data_file + '.data', 'wb'))

    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def cmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para, cv_type):
    dict_perf = {}
    max_aupr, aupr_opt = 0, []
    for d in [50, 100]:
        dict_perf[d] = {}
        for x in np.arange(-2, -1):
            dict_perf[d][x] = {}
            for y in np.arange(-3, -2):
                dict_perf[d][x][y] = {}
                for z in np.arange(-3, -2):
                    tic = time.clock()
                    model = CMF(K=d, lambda_l=2**(x), lambda_d=2**(y), lambda_t=2**(z), max_iter=30)
                    cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + " CV: " + cv_type + "\n" +\
                        str(model)
                    print(cmd)
                    aupr_vec, auc_vec = train(model, cv_data, X, D, T, cv_type)
                    if cv_type != 'loo':
                        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                    else:
                        aupr_avg, aupr_conf, auc_avg, auc_conf = aupr_vec, 0., auc_vec, 0.
                    print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic))
                    if aupr_avg > max_aupr:
                        max_aupr = aupr_avg
                        aupr_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
                    dict_perf[d][x][y][z] = (aupr_vec, auc_vec)

    data_file = dataset + "_" + str(cvs) + "_" + cv_type + '_' + str(model)
    pickle.dump(dict_perf, open('results/' + data_file + '.data', 'wb'))

    cmd = "Optimal parameter setting:\n%s\n" % aupr_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (aupr_opt[1], aupr_opt[2], aupr_opt[3], aupr_opt[4])
    print(cmd)


def nnkronsvm_cv_eval(method, dataset, cv_data, X, D, T, cvs, para, cv_type):
    print(para)
    dict_perf = {}
    max_aupr, aupr_opt = 0, []
    for C in [.0001, .001, .01, .1, 1., 10., 100., 1000]:
        print(C)
        tic = time.clock()
        model = NNKronSVM(C=C, NbNeg=para['NbNeg'], NegNei=para['NegNei'],
                          PosNei=para['PosNei'], dataset=dataset)
        cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + " CV: " + cv_type + "\n" +\
            str(model)
        print(cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T, cv_type)
        if cv_type != 'loo':
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        else:
            aupr_avg, aupr_conf, auc_avg, auc_conf = aupr_vec, 0., auc_vec, 0.
        print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic))
        if aupr_avg > max_aupr:
            max_aupr = aupr_avg
            aupr_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
        dict_perf[C] = (aupr_vec, auc_vec)

    data_file = dataset + "_" + str(cvs) + "_" + cv_type + '_' + str(model)
    pickle.dump(dict_perf, open('results/' + data_file + '.data', 'wb'))

    cmd = "Optimal parameter setting:\n%s\n" % aupr_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (aupr_opt[1], aupr_opt[2], aupr_opt[3], aupr_opt[4])
    print(cmd)
