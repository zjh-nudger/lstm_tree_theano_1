import numpy as np
import theano
import theano.tensor as T
from theano import config
import timeit

class DataTriplet(object):

    def __init__(self, data_list=None, label_vectors=None, label_alphabet_list=None):
        self.training_data = []
        self.training_data_label = []
        self.dev_data = []
        self.dev_data_label = []
        self.test_data = []
        self.test_data_label = []
        self.label_alphabet_list = label_alphabet_list

        if data_list is not None:
            self.training_data = [x for x in data_list[0]]
            self.dev_data = [x for x in data_list[1]]
            self.test_data = [x for x in data_list[2]]
        if label_vectors is not None:
            self.training_data_label = [x for x in label_vectors[0]]
            self.dev_data_label = [x for x in label_vectors[1]]
            self.test_data_label = [x for x in label_vectors[2]]

    def _check_num_rows(self, data_list):
        num_rows = [x.shape[0] for x in data_list]
        assert(all(x == num_rows[0] for x in num_rows))

    def assert_data_same_length(self):
        assert(len(self.training_data) == len(self.dev_data))    
        assert(len(self.test_data) == len(self.dev_data))    

        assert(len(self.training_data_label) == len(self.dev_data_label))    
        assert(len(self.test_data_label) == len(self.dev_data_label))    

        #self._check_num_rows(self.training_data)
        #self._check_num_rows(self.dev_data)
        #self._check_num_rows(self.test_data)

        self._check_num_rows(self.training_data_label)
        self._check_num_rows(self.dev_data_label)
        self._check_num_rows(self.test_data_label)

    def num_input_variables(self):
        self.assert_data_same_length()
        return len(self.training_data)

    def input_dimensions(self):
        return [x.shape[1] for x in self.training_data]

    def num_output_variables(self):
        self.assert_data_same_length()
        return len(self.training_data_label)

    def output_dimensions(self):
        return [len(x) for x in self.label_alphabet_list]

    def training_data_and_label_list(self):
        return self.training_data + self.training_data_label

    def dev_data_and_label_list(self):
        return self.dev_data + self.dev_data_label

    def test_data_and_label_list(self):
        return self.test_data + self.test_data_label


class Trainer(object):

    def train_minibatch(self, minibatch_size, n_epochs, training_data, dev_data, test_data):
        """Train minibatch with one output

        training_data should be a list of [X1, X2, ... Xn,Y]
        """
        data_triplet = DataTriplet()

        data_triplet.training_data.extend(training_data[:-1])
        data_triplet.training_data_label.append(training_data[-1])

        data_triplet.dev_data.extend(dev_data[:-1])
        data_triplet.dev_data_label.append(dev_data[-1])

        data_triplet.test_data.extend(test_data[:-1])
        data_triplet.test_data_label.append(test_data[-1])

        return self.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)


    def train_minibatch_triplet(self, minibatch_size, n_epochs):
        """Train with minibatch

        The early stoping is on the last output variable accuracy only.
        This is crazy stupid ugly solution but it should be ok. 
        """

        patience = 5000
        patience_increase = 2.5 # wait this much longer when a new best is found
        improvement_threshold = 1.0#  0.9975

        n_train_batches = self.num_training_data / minibatch_size
        validation_frequency = min(n_train_batches / 4, patience / 2)
    
        done_looping = False
        epoch = 0
        best_dev_acc = 0.0
        best_dev_iteration = 0
        best_test_acc = 0.0
        total_cost = 0.0
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
                iteration = (epoch - 1) * n_train_batches  + minibatch_index
                start_time = timeit.default_timer()
                #misc = self.misc_function(minibatch_index, minibatch_size)
                #print misc

                c = self.train_function(minibatch_index, minibatch_size)
                if np.isnan(c):
                    print 'NaN found at batch %s after seeing %s samples' % \
                            (minibatch_index, iteration * minibatch_size)
                    done_looping =True
                    break
                total_cost += c
                end_time = timeit.default_timer()
                if (iteration + 1) % validation_frequency == 0:
                    num_samples_seen = iteration * minibatch_size
                    average_cost = total_cost / num_samples_seen
                    print 'TRAIN: iteration %s : average cost =%s' % \
                            (iteration, average_cost)

                    dev_data = self.data_triplet.dev_data_and_label_list()
                    dev_accuracy, c = self.eval_function_dev(*dev_data)
                    print 'DEV: iteration %s : accuracy = %s ; cost =%s' % \
                            (iteration, dev_accuracy, c)

                    test_data = self.data_triplet.test_data_and_label_list()
                    test_accuracy, c = self.eval_function_test(*test_data)
                    print 'TEST: iteration %s : accuracy = %s ; cost =%s' % \
                            (iteration, test_accuracy, c)
                    if dev_accuracy > best_dev_acc:
                        if dev_accuracy * improvement_threshold > best_dev_acc:
                            patience = max(patience, 
                                    iteration * patience_increase)
                        best_dev_acc = dev_accuracy
                        best_dev_iteration = iteration
                        best_test_acc = test_accuracy    
                if patience <= iteration:
                    done_looping = True
                    break

        return best_dev_iteration, float(best_dev_acc), float(best_test_acc)

class AdagradTrainer(Trainer):

    def __init__(self, model, cost_function, learning_rate, lr_smoother, 
            data_triplet, make_givens_fn=None):
        self.model = model
        self.cost_function = cost_function 
        self.learning_rate = learning_rate
        self.lr_smoother = lr_smoother
        self.data_triplet = data_triplet

        print 'Taking gradient...'
        #self.gparams = [T.grad(cost=cost_function, wrt=x) 
                #for x in self.model.params]
        self.gparams = T.grad(cost_function, self.model.params)

        #self.gparams = [\
        #        T.maximum(-5, T.minimum(5, T.grad(cost=cost_function, wrt=x))) 
        #        for x in self.model.params]
        self.sum_gradient_squareds = [
                theano.shared(value=np.zeros(param.get_value().shape).\
                        astype(config.floatX), borrow=True) 
                for param in self.model.params]

        adagrad_rates = [learning_rate / (lr_smoother + T.sqrt(sgs)) 
                for sgs in self.sum_gradient_squareds]

        self.sgs_updates = [(sgs, sgs + T.square(gparam)) 
            for sgs, gparam in zip(self.sum_gradient_squareds, self.gparams)]


        self.param_updates = [(param, param - adagrad_rate * gparam) 
                for param, gparam, adagrad_rate in 
                zip(self.model.params, self.gparams, adagrad_rates)]
        self.train_function = None
        self.dev_eval_function = None
        self.test_eval_function = None

        data_triplet.assert_data_same_length()
        assert(len(self.model.input) == data_triplet.num_input_variables())
        assert(len(self.model.output) == data_triplet.num_output_variables())

        index = T.lscalar() # index to minibatch
        minibatch_size = T.lscalar() # index to minibatch
        T_training_data = [theano.shared(x, borrow=True) 
                for x in data_triplet.training_data]
        T_training_data_label = [theano.shared(x, borrow=True) 
                for x in data_triplet.training_data_label]
        self.num_training_data = len(data_triplet.training_data_label[0])

        givens = {}
        start_idx = index * minibatch_size
        end_idx = (index + 1) * minibatch_size
        if make_givens_fn is not None:
            make_givens_fn(givens, self.model.input, 
                    T_training_data, start_idx, end_idx)
        else:
            for i, input_var in enumerate(self.model.input):
                givens[input_var] = T_training_data[i][start_idx:end_idx]

        for i, output_var in enumerate(self.model.output):
            givens[output_var] = T_training_data_label[i][start_idx:end_idx]

        print 'Compiling training function...'
        self.train_function = theano.function(
                inputs=[index, minibatch_size],
                outputs=self.cost_function,
                updates=self.sgs_updates + self.param_updates,
                givens=givens,
                on_unused_input='warn')

        #ugly crap going on to compute accuracy on the last output variable only
        accuracy = T.mean(T.eq(self.model.output[-1], self.model.predict[-1]))
        self.eval_function_dev = \
                theano.function(inputs=self.model.input + self.model.output, 
                        outputs=[accuracy, self.cost_function],
                        on_unused_input='warn')
        self.eval_function_test = self.eval_function_dev
        print 'Adagrad finished compiling...'

        #self.misc_function = theano.function(
                #inputs=[index, minibatch_size],
                #outputs=self.model.layers[2].input[0],
                #givens=givens)

    def reset(self):
        for sgs in self.sum_gradient_squareds:
            sgs.set_value(np.zeros(sgs.get_value().shape, dtype=theano.config.floatX))


class SGDTrainer(Trainer):

    def __init__(self, model, cost_function, learning_rate):
        self.model = model
        self.cost_function = cost_function 
        self.learning_rate = learning_rate

        self.gparams = [T.grad(cost=cost_function, wrt=x) for x in self.model.params]

