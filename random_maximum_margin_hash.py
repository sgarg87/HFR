import sklearn.svm as skl_svm
import numpy as np


class RandomMaximumMarginHash:

    def compute_hashcode_bit(self, K, Kr, subset1, subset2, reg=1.0, max_iter=-10000):

        n = K.shape[0]
        superset_size = K.shape[1]
        assert superset_size == (subset1.size+subset2.size)

        assert Kr is not None

        labels = np.zeros(superset_size, dtype=np.int)
        labels[subset1] = -1
        labels[subset2] = 1
        # print 'labels', labels

        svm_clf_obj = skl_svm.SVC(
            C=reg,
            kernel='precomputed',
            probability=False,
            verbose=False,
            random_state=0,
            max_iter=max_iter,
            # class_weight='balanced'
        )

        svm_clf_obj.fit(
            Kr,
            labels
        )

        inferred_labels = svm_clf_obj.predict(K)
        z = np.zeros(n, dtype=np.bool)
        z[np.where(inferred_labels == 1)] = 1

        return z
