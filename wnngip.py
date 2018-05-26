
'''
[1] van Laarhoven, Twan, Sander B. Nabuurs, and Elena Marchiori. "Gaussian interaction profile kernels for predicting drug-target interaction." Bioinformatics 27.21 (2011): 3036-3043.
[2] van Laarhoven, Twan, and Elena Marchiori. "Predicting drug-target interactions for new drug compounds using a weighted nearest neighbor profile." PloS one 8.6 (2013): e66952.

Default Parameters:
    T = 0.7 (the parameter T in Section [2])
    sigma = 1.0
    alpha = 0.5
    gamma = 1.0
'''
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc


class WNNGIP:

    def __init__(self, T=0.7, sigma=1, alpha=0.5, gamma=1.0):
        self.T = T      # the decay parameter
        self.sigma = sigma  # the regularization parameter
        self.alpha = alpha  # the weight parameter used in combining different kernels
        self.gamma = gamma  # the bandwidth of the GIP kernel
        self.preprocess = True

    def preprocess_wnn(self, R, S, train_inx, new_inx, drug=True):
        for d in new_inx:
            ii = np.argsort(S[d, train_inx])[::-1]
            inx = train_inx[ii]
            for i in xrange(inx.size):
                w = self.T**(i)
                if w >= 1e-4:
                    if drug:
                        R[d, :] += w * R[inx[i], :]
                    else:
                        R[:, d] += w * R[:, inx[i]]
                else:
                    break

    def rls_kron_train(self, R, Kd, Kt):
        m, n = R.shape
        ld, vd = np.linalg.eig(Kd)
        lt, vt = np.linalg.eig(Kt)
        vec = ld.reshape((ld.size, 1)) * lt.reshape((1, lt.size))
        vec = vec.reshape((1, vec.size))
        x = vec * (1.0 / (vec + self.sigma))
        y = np.dot(np.dot(vt.T, R.T), vd)
        y = y.reshape((1, y.size))
        z = (x * y).reshape((n, m))  # need to check
        self.predictR = np.dot(np.dot(vd, z.T), vt.T)

    def kernel_combination(self, R, S, new_inx, bandwidth):
        K = self.alpha * S + (1.0 - self.alpha) * rbf_kernel(R, gamma=bandwidth)
        K[new_inx, :] = S[new_inx, :]
        K[:, new_inx] = S[:, new_inx]
        return K

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None, epsilon=0.1):
        R = W * intMat
        m, n = intMat.shape
        x, y = np.where(R > 0)
        # Enforce the positive definite property of similarity matrix
        drugMat = (drugMat + drugMat.T) / 2 + epsilon * np.eye(m)
        targetMat = (targetMat + targetMat.T) / 2 + epsilon * np.eye(n)
        train_drugs = np.array(list(set(x.tolist())), dtype=np.int32)
        train_targets = np.array(list(set(y.tolist())), dtype=np.int32)
        new_drugs = np.array(list(set(xrange(m)) - set(x.tolist())), dtype=np.int32)
        new_targets = np.array(list(set(xrange(n)) - set(y.tolist())), dtype=np.int32)
        drug_bw = self.gamma * m / len(x)
        target_bw = self.gamma * n / len(x)
        Kd = self.kernel_combination(R, drugMat, new_drugs, drug_bw)
        Kt = self.kernel_combination(R.T, targetMat, new_targets, target_bw)
        if self.preprocess is True:
            self.preprocess_wnn(R, drugMat, train_drugs, new_drugs, True)
            self.preprocess_wnn(R, targetMat, train_targets, new_targets, False)
        self.rls_kron_train(R, Kd, Kt)

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return self.predictR[inx[:, 0], inx[:, 1]]

    def evaluation(self, test_data, test_label, intMat, R):
        scores = self.predictR[test_data[:, 0], test_data[:, 1]]
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val, scores

    def predict(self, test_data, R, true_test_data=None):
        ii, jj = test_data[:, 0], test_data[:, 1]
        scores = self.predictR[ii, jj]
        if true_test_data is not None:
            ii, jj = true_test_data[:, 0], true_test_data[:, 1]
            self.pred[ii, jj] = scores
        else:
            self.pred[ii, jj] = scores

    def get_perf(self, intMat):
        pred_ind = np.where(self.pred != np.inf)
        pred_local = self.pred[pred_ind[0], pred_ind[1]]
        test_local = intMat[pred_ind[0], pred_ind[1]]
        prec, rec, thr = precision_recall_curve(test_local, pred_local)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_local, pred_local)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def __str__(self):
        return "Model: RLSWNN, T:%s, sigma:%s, alpha:%s, gamma:%s preprocess:%s" % (self.T, self.sigma, self.alpha, self.gamma, self.preprocess)


class GIP(WNNGIP):

    def __init__(self, T=0.7, sigma=1, alpha=0.5, gamma=1.0):
        self.T = T      # the decay parameter
        self.sigma = sigma  # the regularization parameter
        self.alpha = alpha  # the weight parameter used in combining different kernels
        self.gamma = gamma  # the bandwidth of the GIP kernel
        self.preprocess = False


class NNWNNGIP(WNNGIP):

    def __init__(self, T=0.7, sigma=1, alpha=0.5, gamma=1.0, NN=10):
        self.T = T      # the decay parameter
        self.sigma = sigma  # the regularization parameter
        self.alpha = alpha  # the weight parameter used in combining different kernels
        self.gamma = gamma  # the bandwidth of the GIP kernel
        self.NN = NN
        self.preprocess = True

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None, epsilon=0.1):
        R = W * intMat
        m, n = intMat.shape
        x, y = np.where(R > 0)
        # Enforce the positive definite property of similarity matrix
        drugMat = (drugMat + drugMat.T) / 2 + epsilon * np.eye(m)
        targetMat = (targetMat + targetMat.T) / 2 + epsilon * np.eye(n)
        train_drugs = np.array(list(set(x.tolist())), dtype=np.int32)
        train_targets = np.array(list(set(y.tolist())), dtype=np.int32)
        new_drugs = np.array(list(set(xrange(m)) - set(x.tolist())), dtype=np.int32)
        new_targets = np.array(list(set(xrange(n)) - set(y.tolist())), dtype=np.int32)
        drug_bw = self.gamma * m / len(x)
        target_bw = self.gamma * n / len(x)
        Kd = self.kernel_combination(R, drugMat, new_drugs, drug_bw)
        Kt = self.kernel_combination(R.T, targetMat, new_targets, target_bw)
        if self.preprocess is True:
            self.preprocess_wnn(R, drugMat, train_drugs, new_drugs, True)
            self.preprocess_wnn(R, targetMat, train_targets, new_targets, False)
        self.R, self.Kt, self.Kd = R, Kt, Kd

    # def evaluation(self, test_data, test_label, intMat):
    #     scores = self.predictR[test_data[:, 0], test_data[:, 1]]
    #     prec, rec, thr = precision_recall_curve(test_label, scores)
    #     aupr_val = auc(rec, prec)
    #     fpr, tpr, thr = roc_curve(test_label, scores)
    #     auc_val = auc(fpr, tpr)
    #     return aupr_val, auc_val, scores

    def evaluation(self, test_data, test_label, intMat, R):
        self.predict(test_data, intMat)
        aupr_val, auc_val = self.get_perf(intMat)
        pred_ind = np.where(self.pred != np.inf)
        pred_local = self.pred[pred_ind[0], pred_ind[1]]
        return aupr_val, auc_val, pred_local

    def reduce_kernel(self, K, index):
        size_nei = self.NN + 1 if self.NN + 1 < K.shape[0] else K.shape[0]
        nei = (np.argsort(K[index, :])[::-1])[: size_nei]
        Klocal = K[nei, :]
        Klocal = Klocal[:, nei]
        return Klocal, nei

    def predict(self, test_data, R):
        ii, jj = test_data[:, 0], test_data[:, 1]
        for ind in range(len(ii)):
            ii_test, jj_test = ii[ind], jj[ind]

            Kd_local, mol_nei = self.reduce_kernel(self.Kd, ii_test)
            Kt_local, prot_nei = self.reduce_kernel(self.Kt, jj_test)
            Rlocal = self.R[mol_nei, :]
            Rlocal = Rlocal[:, prot_nei]

            self.rls_kron_train(Rlocal, Kd_local, Kt_local)
            scores = self.predictR[0, 0]
            self.pred[ii_test, jj_test] = scores


class NNGIP(NNWNNGIP):

    def __init__(self, T=0.7, sigma=1, alpha=0.5, gamma=1.0, NN=10):
        self.T = T      # the decay parameter
        self.sigma = sigma  # the regularization parameter
        self.alpha = alpha  # the weight parameter used in combining different kernels
        self.gamma = gamma  # the bandwidth of the GIP kernel
        self.NN = NN
        self.preprocess = False
