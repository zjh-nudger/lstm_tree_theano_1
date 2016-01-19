import numpy as np
import theano
from theano import config
import theano.tensor as T
import tree_util

class LSTM(object):

    def square_weight(self, ndim):
        return self.rng.uniform(
                        low=-np.sqrt(6. / (4 * ndim)),
                        high=np.sqrt(6. / (4 * ndim)),
                        size=(ndim, ndim)
                    ).astype(config.floatX)


    def _init_params(self, rng, dim_proj, n_out=None, W=None, U=None, b=None, 
            num_slices=4):
        self.params = []#a list of paramter variables
        self.input = [] #a list of input variables
        self.output = []#a list of output variables
        self.predict = [] # a list of prediction functions
        self.hinge_loss = None # a function
        self.crossentropy = None # a function

        self.dim_proj = dim_proj
        self.rng = rng
        #LSTM parameters
        if W is None:
            W_values = np.concatenate(
                    [self.square_weight(self.dim_proj) 
                        for x in range(num_slices)], axis=1)#行链接起来
            self.W = theano.shared(W_values, borrow=True)
        else:
            self.W = W

        if U is None:
            U_values = np.concatenate(
                    [self.square_weight(self.dim_proj) 
                        for x in range(num_slices)], axis=1)
            self.U = theano.shared(U_values, borrow=True)
        else:
            self.U = U 

        if b is None:
            b_values = np.zeros((num_slices * self.dim_proj,)).\
                    astype(config.floatX)
            self.b = theano.shared(b_values, borrow=True)
        else:
            self.b = b
        self.params = [self.W, self.U, self.b]

    def _reset(self, rng, num_slices):
        W_values = np.concatenate(
                [self.square_weight(self.dim_proj) 
                    for x in range(num_slices)], axis=1)
        U_values = np.concatenate(
                [self.square_weight(self.dim_proj) 
                    for x in range(num_slices)], axis=1)
        b_values = np.zeros((num_slices * self.dim_proj,)).astype(config.floatX)

        self.W.set_value(W_values)
        self.U.set_value(U_values)
        self.b.set_value(b_values)


class BinaryTreeLSTM(LSTM):
    """Tree LSTM 

    This version is slow and a bit expensive on memory as the theano.scan loop
    passes on all of the time slices with on one time slice changed. 
    I am not sure if theano is doing something smart with it or not. 

    tlstm.py is unuseable because it takes too long to compose a computation
    graph in theano. It takes around 12s per discourse relation, which is 
    far too slow. 

    """

    def __init__(self, rng, dim_proj, n_out=None, W=None, U=None, b=None):
        self._init_params(rng, dim_proj, n_out, W, U, b, 5)#left and right
        word_matrix = T.tensor3(dtype=config.floatX)  # sen_length * batch * dim

        c_mask = T.matrix(dtype=config.floatX)
        node_mask = T.matrix(dtype=config.floatX)
        children = T.tensor3(dtype='int64')

        self.input = [word_matrix, children, c_mask, node_mask]
        n_samples = word_matrix.shape[1]

        self.h, self.c_memory = self.project(word_matrix, children, c_mask)
        all_samples = T.arange(n_samples)

        self.max_pooled_h = (self.h * node_mask[:, :, None]).max(axis=0) 
        self.sum_pooled_h = (self.h * node_mask[:, :, None]).sum(axis=0) 

        self.mean_pooled_h = self.sum_pooled_h / c_mask.sum(axis=0)[:, None]
        num_inner_nodes = c_mask.sum(axis=0).astype('int64')
        num_nodes = num_inner_nodes * 2 + 1
        self.top_h = self.h[num_nodes - 1, all_samples, :]

    def reset(self, rng):
        self._reset(rng, 5)


    def project(self, word_embedding, children, c_mask):
        """

        word_embedding - TxNxd serrated tensor prefilled with word embeddings 
        children - TxNx3 serrated matrix for children list 
        c_mask - TxN mask for varying children list
        """
        nsteps = children.shape[0]
        if word_embedding.ndim == 3:
            n_samples = word_embedding.shape[1]
        else:
            n_samples = 1

        def _step(c, c_m, hidden, c_matrix):
            node_idx = c[:, 0]
            left_child_idx = c[:, 1]
            right_child_idx = c[:, 2]

            all_samples = T.arange(n_samples)
            recursive = \
                    T.dot(hidden[left_child_idx, all_samples, :], self.W) +\
                    T.dot(hidden[right_child_idx, all_samples, :], self.U) +\
                    self.b

            i = T.nnet.sigmoid(_slice(recursive, 0, self.dim_proj))
            f1 = T.nnet.sigmoid(_slice(recursive, 1, self.dim_proj))
            f2 = T.nnet.sigmoid(_slice(recursive, 2, self.dim_proj))
            o = T.nnet.sigmoid(_slice(recursive, 3, self.dim_proj))
            c_prime = T.tanh(_slice(recursive, 4, self.dim_proj))

            new_c = i * c_prime + \
                    f1 * c_matrix[left_child_idx, all_samples,: ] +\
                    f2 * c_matrix[right_child_idx, all_samples,: ]

            new_c_masked = c_m[:,None] * new_c + \
                    (1. - c_m[:, None]) * c_matrix[node_idx, all_samples, :]

            new_h = o * T.tanh(new_c_masked)
            new_h_masked = c_m[:, None] * new_h + \
                    (1. - c_m[:, None]) * hidden[node_idx, all_samples, :]

            return T.set_subtensor(hidden[node_idx, all_samples], new_h_masked), \
                    T.set_subtensor(c_matrix[node_idx, all_samples], new_c_masked)

        rval, updates = theano.scan(_step, sequences=[children, c_mask],
                outputs_info=[word_embedding,
                    T.zeros(word_embedding.shape),],
                n_steps=nsteps)
        return rval[0][-1], rval[1][-1]

    @staticmethod
    def make_givens(givens, input_vec, T_training_data, start_idx, end_idx):
        
        givens[input_vec[0]] = T_training_data[0][:,start_idx:end_idx, :]
        givens[input_vec[1]] = T_training_data[1][:,start_idx:end_idx, :]
        givens[input_vec[2]] = T_training_data[2][:,start_idx:end_idx]
        givens[input_vec[3]] = T_training_data[3][:,start_idx:end_idx]

        givens[input_vec[0+4]] = T_training_data[0+4][:,start_idx:end_idx, :]
        givens[input_vec[1+4]] = T_training_data[1+4][:,start_idx:end_idx, :]
        givens[input_vec[2+4]] = T_training_data[2+4][:,start_idx:end_idx]
        givens[input_vec[3+4]] = T_training_data[3+4][:,start_idx:end_idx]

        #for i, input_var in enumerate(input_vec[10:]):
            #givens[input_var] = T_training_data[i][start_idx:end_idx]

class SerialLSTM(LSTM):

    def __init__(self, rng, dim_proj, n_out=None, W=None, U=None, b=None):
        self._init_params(rng, dim_proj, n_out, W, U, b, 4)

        X = T.tensor3('x', dtype=config.floatX) 
        mask = T.matrix('mask', dtype=config.floatX)
        self.input = [X, mask]
        n_samples = X.shape[1]

        self.h = self.project(X, mask)

        self.max_pooled_h = (self.h * mask[:, :, None]).max(axis=0) 
        self.sum_pooled_h = (self.h * mask[:, :, None]).sum(axis=0) 
        self.mean_pooled_h = self.sum_pooled_h / mask.sum(axis=0)[:, None]
        self.top_h = self.h[mask.sum(axis=0).astype('int64') - 1,
                T.arange(n_samples), :]


    def reset(self, rng):
        self._reset(rng, 4)

    def project(self, embedding_series, mask):
        nsteps = embedding_series.shape[0]
        if embedding_series.ndim == 3:
            n_samples = embedding_series.shape[1]
        else:
            n_samples = 1

        def _step(m_, x_, h_, c_):
            # x_ is actually x * W so we don't multiply again
            preact = T.dot(h_, self.U) + x_

            i = T.nnet.sigmoid(_slice(preact, 0, self.dim_proj))
            f = T.nnet.sigmoid(_slice(preact, 1, self.dim_proj))
            o = T.nnet.sigmoid(_slice(preact, 2, self.dim_proj))
            c = T.tanh(_slice(preact, 3, self.dim_proj))

            c = f * c_ + i * c
            # if the sequence is shorter than the max length, then pad 
            # it with the old stuff for c and h
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

        previous_state = T.dot(embedding_series, self.W) + self.b

        rval, updates = theano.scan(_step, sequences=[mask, previous_state], 
                outputs_info=[T.alloc(np_floatX(0.), n_samples, self.dim_proj), 
                    T.alloc(np_floatX(0.), n_samples, self.dim_proj)], 
                n_steps=nsteps)
        return rval[0]

    @staticmethod
    def make_givens(givens, input_vec, T_training_data, start_idx, end_idx):
        # first arg embedding and mask
        givens[input_vec[0]] = T_training_data[0][:,start_idx:end_idx, :]
        givens[input_vec[1]] = T_training_data[1][:,start_idx:end_idx]

        # second arg embedding and mask
        givens[input_vec[2]] = T_training_data[2][:,start_idx:end_idx, :]
        givens[input_vec[3]] = T_training_data[3][:,start_idx:end_idx]

        # the rest if there is more e.g. input for MOE
        for i, input_var in enumerate(input_vec[4:]):
            givens[input_var] = T_training_data[i][start_idx:end_idx]

def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


def prep_srm_arg(relation_list, arg_pos, wbm, max_length, ignore_OOV=True):
    assert arg_pos == 1 or arg_pos == 2
    n_samples = len(relation_list)
    x = np.zeros((max_length, n_samples)).astype('int64')
    x_mask = np.zeros((max_length, n_samples)).astype(config.floatX)
    for i, relation in enumerate(relation_list):
        indices = wbm.index_tokens(relation.arg_tokens(arg_pos), ignore_OOV)
        sequence_length = min(max_length, len(indices))
        x[:sequence_length, i] = indices[:sequence_length]
        x_mask[:sequence_length, i] = 1.
    embedding_series = \
        wbm.wm[x.flatten()].\
            reshape([max_length, n_samples, wbm.num_units]).\
            astype(config.floatX)
    return embedding_series, x_mask


def prep_tree_srm_arg(relation_list, arg_pos, wbm, max_length, 
        all_left_branching=False):
    assert arg_pos == 1 or arg_pos == 2
    n_samples = len(relation_list)
    w_indices = np.zeros((2 * max_length, n_samples)).astype('int64')
    c_mask = np.zeros((max_length, n_samples), dtype=config.floatX)
    node_mask = np.zeros((2 * max_length, n_samples), dtype=config.floatX)
    #children = np.zeros((max_length, n_samples, 2), dtype='int64')
    children = np.zeros((n_samples, max_length, 3), dtype='int64')
    for i, relation in enumerate(relation_list):
        if all_left_branching:
            parse_tree = tree_util.left_branching_tree(relation, arg_pos)
        else:
            parse_tree = tree_util.find_parse_tree(relation, arg_pos)
            if len(parse_tree.leaves()) == 0:
                parse_tree = tree_util.left_branching_tree(relation, arg_pos)
        indices = wbm.index_tokens(parse_tree.leaves(), ignore_OOV=False)

        sequence_length = min(max_length, len(indices))
        w_indices[:sequence_length, i] = indices[:sequence_length]

        ordering_matrix, num_leaves = tree_util.reverse_toposort(parse_tree)
        num_nodes = min(2 * max_length, ordering_matrix.shape[0])
        print num_leaves, num_nodes
        #assert(num_nodes >= num_leaves)
        if num_nodes > num_leaves:
            num_inner_nodes = num_nodes - num_leaves
            children[i, :num_inner_nodes, :] = ordering_matrix[num_leaves:num_nodes, :]
            c_mask[:num_inner_nodes, i] = 1.
            node_mask[num_leaves:num_nodes, i] = 1.
    children = np.swapaxes(children, 0, 1)
    embedding_series = \
        wbm.wm[w_indices.flatten()].\
            reshape([max_length * 2, n_samples, wbm.num_units]).\
            astype(config.floatX)
    return embedding_series, children, c_mask, node_mask

def prep_serrated_matrix_relations(relation_list, wbm, max_length):
    arg1_srm, arg1_mask = prep_srm_arg(relation_list, 1, wbm, max_length)
    arg2_srm, arg2_mask = prep_srm_arg(relation_list, 2, wbm, max_length)
    return (arg1_srm, arg1_mask, arg2_srm, arg2_mask)

def _check_masks(word_mask, c_mask):
    """Make sure that the two mask matrices are well-formed

    word_mask and c_mask are T x N 
    We have to check that 0<= w_x + c_m <= 1. They can't both be 1.
    """
    check_sum = word_mask + c_mask
    assert(np.all(0 <= check_sum) and np.all(check_sum <= 1))

def prep_tree_lstm_serrated_matrix_relations(relation_list, wbm, max_length):
    arg1_srm, arg1_children, arg1_c_mask, arg1_node_mask = \
            prep_tree_srm_arg(relation_list, 1, wbm, max_length)
    arg2_srm, arg2_children, arg2_c_mask, arg2_node_mask = \
            prep_tree_srm_arg(relation_list, 2, wbm, max_length)
    return (arg1_srm, arg1_children, arg1_c_mask, arg1_node_mask, \
            arg2_srm, arg2_children, arg2_c_mask, arg2_node_mask)


def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)
    
if __name__=='__main__':
    prep_tree_srm_arg()
