import unittest
import numpy as np
import theano
import theano.tensor as T



class TestIBCCIterable(unittest.TestCase):

#    def setUp(self):


    def test_theano_eq(self):
        a = np.array([0, 1, 0])
        b = T.vector('b')
        true_zero_loc = T.eq(b, 0)
        f = theano.function(inputs=[b], outputs=true_zero_loc)
        c = f(a)
        self.assertTrue(np.allclose(c, np.array([1, 0, 1])))

    def test_theano_neq(self):
        a = np.array([0, 1, 0])
        b = T.vector('b')
        true_nonzero_loc = T.neq(b, 0)
        f = theano.function(inputs=[b], outputs=true_nonzero_loc)
        c = f(a)
        self.assertTrue(np.allclose(c, np.array([0, 1, 0])))

    def test_theano_tp(self):
        true_signal = np.array([0, 1, 0])
        est_signal = np.array([0, 1, 1])
        correct_tp = np.sum((true_signal == 1) * (est_signal == 1))
        true_b = T.vector('true_b')
        est_b = T.vector('est_b')

        true_nonzero_loc = T.neq(true_b, 0)
        est_nonzero_loc = T.neq(est_b, 0)

        tp = T.sum(true_nonzero_loc * est_nonzero_loc)
        f = theano.function(inputs=[true_b, est_b], outputs=tp)
        c = f(true_signal, est_signal)
        self.assertTrue(np.allclose(c, correct_tp))

    def test_theano_fp(self):
        true_signal = np.array([0, 1, 0])
        est_signal = np.array([0, 1, 1])
        correct_fp = np.sum((true_signal == 0) * (est_signal == 1))
        true_b = T.vector('true_b')
        est_b = T.vector('est_b')

        true_zero_loc = T.eq(true_b, 0)
        est_nonzero_loc = T.neq(est_b, 0)

        fp = T.sum(true_zero_loc * est_nonzero_loc)
        f = theano.function(inputs=[true_b, est_b], outputs=fp)
        c = f(true_signal, est_signal)
        self.assertTrue(np.allclose(c, correct_fp))

    def test_theano_fn(self):
        true_signal = np.array([0, 1, 0])
        est_signal = np.array([0, 1, 1])
        correct_fn = np.sum((true_signal == 1) * (est_signal == 0))
        true_b = T.vector('true_b')
        est_b = T.vector('est_b')

        true_nonzero_loc = T.neq(true_b, 0)
        est_zero_loc = T.eq(est_b, 0)

        fn = T.sum(true_nonzero_loc * est_zero_loc)
        f = theano.function(inputs=[true_b, est_b], outputs=fn)
        c = f(true_signal, est_signal)
        self.assertTrue(np.allclose(c, correct_fn))

    def test_theano_f_measure(self):
        true_signal = np.array([0, 1, 0])
        est_signal = np.array([0, 1, 1])
        correct_tp = np.sum((true_signal == 1) * (est_signal == 1))
        correct_fp = np.sum((true_signal == 0) * (est_signal == 1))
        correct_fn = np.sum((true_signal == 1) * (est_signal == 0))
        correct_precision = correct_tp / (correct_tp + correct_fp)
        correct_recall = correct_tp / (correct_tp + correct_fn)
        if correct_precision + correct_recall > 0:
            correct_f_measure = 2 * correct_precision * correct_recall / (correct_precision + correct_recall)
        else:
            correct_f_measure = 0

        true_b = T.vector('true_b')
        est_b = T.vector('est_b')

        true_zero_loc = T.eq(true_b, 0)
        true_nonzero_loc = T.neq(true_b, 0)
        est_zero_loc = T.eq(est_b, 0)
        est_nonzero_loc = T.neq(est_b, 0)

        tp = T.sum(true_nonzero_loc * est_nonzero_loc)
        fp = T.sum(true_zero_loc * est_nonzero_loc)
        fn = T.sum(true_nonzero_loc * est_zero_loc)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f_meas = T.switch(T.gt(precision + recall, 0), 2 * (precision * recall / (precision + recall)), 0)

        f = theano.function(inputs=[true_b, est_b], outputs=f_meas)
        c = f(true_signal, est_signal)
        self.assertTrue(np.allclose(c, correct_f_measure))

    def test_theano_f_measure_strange_numbers(self):
        thr_lambda = 0.2
        true_signal = np.random.normal(0, 3, (100, 1))
        true_signal = np.sign(true_signal) * np.maximum(abs(true_signal) - thr_lambda, np.zeros_like(true_signal))
        est_signal = np.random.normal(0, 3, (100, 1))
        est_signal = np.sign(est_signal) * np.maximum(abs(est_signal) - thr_lambda, np.zeros_like(est_signal))


        correct_tp = np.sum((true_signal != 0) * (est_signal != 0))
        correct_fp = np.sum((true_signal == 0) * (est_signal != 0))
        correct_fn = np.sum((true_signal != 0) * (est_signal == 0))
        correct_precision = correct_tp / (correct_tp + correct_fp)
        correct_recall = correct_tp / (correct_tp + correct_fn)
        if correct_precision + correct_recall > 0:
            correct_f_measure = 2 * correct_precision * correct_recall / (correct_precision + correct_recall)
        else:
            correct_f_measure = 0


        # Bayesian Lista code
        true_zero_loc = true_signal == 0
        true_nonzero_loc = true_signal != 0
        est_zero_loc = est_signal == 0
        est_nonzero_loc = est_signal != 0

        tp = np.sum(true_nonzero_loc * est_nonzero_loc)
        fp = np.sum(true_zero_loc * est_nonzero_loc)
        fn = np.sum(true_nonzero_loc * est_zero_loc)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision + recall > 0:
            bayes_lista_f_measure = 2 * (precision * recall / (precision + recall))
        else:
            bayes_lista_f_measure = 0





        true_b = T.vector('true_b')
        est_b = T.vector('est_b')

        true_zero_loc = T.eq(true_b, 0)
        true_nonzero_loc = T.neq(true_b, 0)
        est_zero_loc = T.eq(est_b, 0)
        est_nonzero_loc = T.neq(est_b, 0)

        tp = T.sum(true_nonzero_loc * est_nonzero_loc)
        fp = T.sum(true_zero_loc * est_nonzero_loc)
        fn = T.sum(true_nonzero_loc * est_zero_loc)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f_meas = T.switch(T.gt(precision + recall, 0), 2 * (precision * recall / (precision + recall)), 0)

        f = theano.function(inputs=[true_b, est_b], outputs=f_meas)
        freq_lista_f_measure = f(true_signal, est_signal)
        self.assertTrue(np.allclose(freq_lista_f_measure, correct_f_measure))
        self.assertTrue(np.allclose(bayes_lista_f_measure, correct_f_measure))



    def test_theano_f_measure_strange_numbers(self):
        true_signal = np.array([1e-300, 1.0, 1e-400, 0.0, 2.0-2.0])
        est_signal = np.array([1e-350, 1.0, 1.0, 1e-600, 1.0])
        correct_tp = np.sum((true_signal != 0) * (est_signal != 0))
        correct_fp = np.sum((true_signal == 0) * (est_signal != 0))
        correct_fn = np.sum((true_signal != 0) * (est_signal == 0))
        correct_precision = correct_tp / (correct_tp + correct_fp)
        correct_recall = correct_tp / (correct_tp + correct_fn)
        if correct_precision + correct_recall > 0:
            correct_f_measure = 2 * correct_precision * correct_recall / (correct_precision + correct_recall)
        else:
            correct_f_measure = 0

        true_b = T.vector('true_b')
        est_b = T.vector('est_b')

        true_zero_loc = T.eq(true_b, 0)
        true_nonzero_loc = T.neq(true_b, 0)
        est_zero_loc = T.eq(est_b, 0)
        est_nonzero_loc = T.neq(est_b, 0)

        tp = T.sum(true_nonzero_loc * est_nonzero_loc)
        fp = T.sum(true_zero_loc * est_nonzero_loc)
        fn = T.sum(true_nonzero_loc * est_zero_loc)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f_meas = T.switch(T.gt(precision + recall, 0), 2 * (precision * recall / (precision + recall)), 0)

        f = theano.function(inputs=[true_b, est_b], outputs=f_meas)
        c = f(true_signal, est_signal)
        self.assertTrue(np.allclose(c, correct_f_measure))






if __name__ == '__main__':
    unittest.main()