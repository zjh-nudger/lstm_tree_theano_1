from nltk import Tree
from lstm import LSTM

import theano
from theano import config
import theano.tensor as T
import tree_util

class BinaryForestLSTM(LSTM):

    def __init__(self, data, rng, wbm, 
            X_list=None, n_out=None, W=None, U=None, b=None):
        dim_proj = wbm.num_units
        self._init_params(rng, dim_proj, n_out, W, U, b, 5)
        self.wbm = wbm
        self.trees = [self.make_tree(tree, wbm) for tree in data]
        if X_list is not None:
            indices = X_list[0]
            self.input = X_list
        else:
            indices = T.lvector()
            self.input = [indices]
        self.all_max_pooled_h = T.concatenate([t.max_pooled_h for t in self.trees])
        self.all_mean_pooled_h = T.concatenate([t.mean_pooled_h for t in self.trees])
        self.all_sum_pooled_h = T.concatenate([t.sum_pooled_h for t in self.trees])
        self.all_top_h = T.concatenate([t.h[None,:] for t in self.trees], axis=0)

        self.max_pooled_h = self.all_max_pooled_h[indices, :]
        self.mean_pooled_h = self.all_mean_pooled_h[indices, :]
        self.sum_pooled_h = self.all_sum_pooled_h[indices, :]
        self.top_h = self.all_top_h[indices, :]

    def copy(self, data):
        model = BinaryForestLSTM(data, None, 
                self.wbm, X_list=self.input, W=self.W, U=self.U, b=self.b)
        return model

    def reset(self, rng):
        self._reset(rng, 5)

    def make_tree(self, t, wbm):
        hidden_variables = []
        root = self._recurs_make_tree(t, wbm, hidden_variables)
        hidden_matrix = T.concatenate(hidden_variables, axis=1)
        root.max_pooled_h = hidden_matrix.max(axis=1)[None,:]
        root.sum_pooled_h = hidden_matrix.sum(axis=1)[None,:]
        root.mean_pooled_h = hidden_matrix.mean(axis=1)[None,:]
        return root

    def _recurs_make_tree(self, t, wbm, hidden_variables):
        if not isinstance(t, Tree):
            n = TLSTMNode(self, t, left_child=None, right_child=None)
            return n

        left = self._recurs_make_tree(t[0], wbm, hidden_variables)
        right = self._recurs_make_tree(t[1], wbm, hidden_variables)
        new_root = TLSTMNode(self, None, left, right)
        hidden_variables.append(new_root.h[:,None])
        return new_root

    @staticmethod
    def make_givens(givens, input_vec, T_training_data, start_idx, end_idx):
        givens[input_vec[0]] = T_training_data[0][start_idx:end_idx]

class TLSTMNode(object):

    def __init__(self, model, word, left_child=None, right_child=None):
        self.left = left_child
        self.right = right_child
        self.W = model.W
        self.U = model.U
        self.b = model.b
        self.wbm = model.wbm
        self.dim_proj = model.dim_proj

        if self.left is None and self.right is None:
            self.c = T.zeros(self.dim_proj)
            #self.embedding = self.wbm.get_embedding(word)
            self.h = self.wbm.get_embedding(word)
        else:
            preact = T.dot(self.left.h, self.W) +\
                    T.dot(self.right.h, self.U) + self.b

            #i = T.nnet.sigmoid(_slice(preact, 0, self.dim_proj))
            #f1 = T.nnet.sigmoid(_slice(preact, 1, self.dim_proj))
            #f2 = T.nnet.sigmoid(_slice(preact, 2, self.dim_proj))
            #o = T.nnet.sigmoid(_slice(preact, 3, self.dim_proj))
            #c_prime = T.tanh(_slice(preact, 4, self.dim_proj))
            i = T.nnet.sigmoid(preact[0:self.dim_proj])
            f1 = T.nnet.sigmoid(preact[self.dim_proj:self.dim_proj*2])
            f2 = T.nnet.sigmoid(preact[(self.dim_proj*2):(self.dim_proj*3)])
            o = T.nnet.sigmoid(preact[self.dim_proj*3:self.dim_proj*4])
            c_prime = T.tanh(preact[self.dim_proj*4:self.dim_proj*5])

            self.c = i * c_prime + f1 * self.left.c + f2 * self.right.c
            self.h = (o * T.tanh(self.c))

def prep_tree_arg(relation_list, arg_pos, all_left_branching=False):
    parse_trees = []
    for i, relation in enumerate(relation_list):
        if all_left_branching:
            parse_tree = tree_util.left_branching_tree(relation, arg_pos)
        else:
            parse_tree = tree_util.find_parse_tree(relation, arg_pos)
            print parse_tree
            if len(parse_tree.leaves()) == 0:
                print 'use left branching tree because parse is empty'
                parse_tree = tree_util.left_branching_tree(relation, arg_pos)
            else:
                parse_tree = tree_util.binarize_tree(parse_tree)
            print parse_tree
        parse_trees.append(parse_tree)
    return parse_trees

def prep_trees(relation_list, all_left_branching=False):
    arg1_trees = prep_tree_arg(relation_list, 1, all_left_branching)
    arg2_trees = prep_tree_arg(relation_list, 2, all_left_branching)
    return (arg1_trees, arg2_trees)
