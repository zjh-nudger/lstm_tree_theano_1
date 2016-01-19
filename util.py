import numpy as np
import scipy as sp
from cognitive_disco.data_reader import extract_implicit_relations

def _sparse_featurize_relation_list(relation_list, ff_list, alphabet=None):
	if alphabet is None:
		alphabet = {}
		grow_alphabet = True
	else:
		grow_alphabet = False
	feature_vectors = []
	print 'Applying feature functions...'
	for relation in relation_list:
		feature_vector_indices = []
		for ff in ff_list:
			feature_vector = ff(relation)
			for f in feature_vector:
				if grow_alphabet and f not in alphabet:
					alphabet[f] = len(alphabet)
				if f in alphabet:
					feature_vector_indices.append(alphabet[f])
		feature_vectors.append(feature_vector_indices)

	print 'Creating feature sparse matrix...'
	feature_matrix = sp.sparse.lil_matrix((len(relation_list), len(alphabet)))
	for i, fv in enumerate(feature_vectors):
		feature_matrix[i, fv] = 1	
	return feature_matrix.tocsr(), alphabet	

def sparse_featurize(relation_list_list, ff_list):
	print 'Featurizing...'
	data_list = []
	alphabet = None
	for relation_list in relation_list_list:
		data, alphabet = _sparse_featurize_relation_list(relation_list, ff_list, alphabet)
		data_list.append(data)
	return (data_list, alphabet)	

def compute_mi(feature_matrix, label_vector):
	"""Compute mutual information of each feature 

	"""
	num_labels = np.max(label_vector) + 1
	num_features = feature_matrix.shape[1]
	num_rows = feature_matrix.shape[0]
	total = num_rows + num_labels

	c_y = np.zeros(num_labels)
	for l in label_vector:
		c_y[l] += 1.0
	c_y += 1.0

	c_x_y = np.zeros((num_features, num_labels)) 
	c_x = np.zeros(num_features)
	for i in range(num_rows):
		c_x_y[:, label_vector[i]] += feature_matrix[i, :]
		c_x += feature_matrix[i, :]
	c_x_y += 1.0
	c_x += 1.0

	c_x_c_y = np.outer(c_x, c_y)
	c_not_x_c_y = np.outer((total - c_x), c_y)
	c_not_x_y = c_y - c_x_y

	inner = c_x_y / total * np.log(c_x_y * total / c_x_c_y) + \
			c_not_x_y / total * np.log(c_not_x_y * total / c_not_x_c_y) 
	mi_x = inner.sum(1)	
	return mi_x

def prune_feature_matrices(feature_matrices, mi, num_features):
	sorted_indices = mi.argsort()[-num_features:]
	return [x[:, sorted_indices] for x in feature_matrices]

class BrownDictionary(object):

	def __init__(self):
		self.word_to_brown_mapping = {}
		self.num_clusters = 0
		brown_cluster_file_name  = 'brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt'
		#brown_cluster_file_name  = 'brown-rcv1.clean.tokenized-CoNLL03.txt-c320-freq1.txt'
		#brown_cluster_file_name  = 'brown-rcv1.clean.tokenized-CoNLL03.txt-c100-freq1.txt'
		self._load_brown_clusters('resources/%s' % brown_cluster_file_name)

	def _load_brown_clusters(self, path):
		try:
			lexicon_file = open(path)
		except:
			print 'fail to load brown cluster data'
		cluster_set = set()
		for line in lexicon_file:
			cluster_assn, word, _ = line.split('\t')
			if cluster_assn not in cluster_set:
				cluster_set.add(cluster_assn)
			self.word_to_brown_mapping[word] = len(cluster_set) - 1
		self.num_clusters = len(cluster_set)

	def _get_brown_cluster_bag(self, tokens):
		bag = set()
		for token in tokens:
			if token in self.word_to_brown_mapping:
				cluster_assn = self.word_to_brown_mapping[token]
				if cluster_assn not in bag:
					bag.add(cluster_assn)
		return bag

	def get_brown_sparse_matrices_relations(self, relations):
		X1 = sp.sparse.csr_matrix((len(relations), self.num_clusters),dtype=float)	
		X2 = sp.sparse.csr_matrix((len(relations), self.num_clusters),dtype=float)	
		for i, relation in enumerate(relations):
			bag1 = self._get_brown_cluster_bag(relation.arg_tokens(1))
			for cluster in bag1:
				X1[i, cluster] = 1.0
			bag2 = self._get_brown_cluster_bag(relation.arg_tokens(2))
			for cluster in bag2:
				X2[i, cluster] = 1.0
		return (X1, X2)

	def get_brown_matrices_data(self, relation_list_list, use_sparse):
		"""Extract sparse 

		For each directory, returns
			(X1, X2, Y) 
			X1 and X2 are sparse matrices from arg1 and arg2 respectively.
			Y is an integer vector of type int32
		"""
		data = []
		alphabet = None
		# load the data
		for relation_list in relation_list_list:
			# turn them into a data matrix
			print 'Making matrices' 
			X1, X2 = self.get_brown_sparse_matrices_relations(relation_list)	
			if not use_sparse:
				X1 = X1.toarray()
				X2 = X2.toarray()
			Y, alphabet = level2_labels(relation_list, alphabet)
			data.append((X1, X2, Y))
		return (data, alphabet)

def label_vectorize(relation_list_list, lf):
	alphabet = {}
	for i, valid_label in enumerate(lf.valid_labels()):
		alphabet[valid_label] = i 

	label_vectors = []
	for relation_list in relation_list_list:
		label_vector = [alphabet[lf.label(x)] for x in relation_list]
		label_vectors.append(np.array(label_vector, np.int64))
	return label_vectors, alphabet

def convert_level2_labels(relations):
	# TODO: this is not enough because we have to exclude some tinay classes
	new_relation_list = []
	for relation in relations:
		split_sense = relation.senses[0].split('.')
		if len(split_sense) >= 2:
			relation.relation_dict['Sense']= ['.'.join(split_sense[0:2])]
			new_relation_list.append(relation)
	return new_relation_list

def level2_labels(relations, alphabet=None):
	if alphabet is None:
		alphabet = {}
		label_set = set()
		for relation in relations:
			label_set.add(relation.senses[0])
		print label_set
		sorted_label = sorted(list(label_set))
		for i, label in enumerate(sorted_label):
			alphabet[label] = i
	label_vector = []
	for relation in relations:
		if relation.senses[0] not in alphabet:
			alphabet[relation.senses[0]] = len(alphabet)
		label_vector.append(alphabet[relation.senses[0]])
	return np.array(label_vector, np.int64), alphabet


if __name__ == '__main__':
	fm = np.array([ [1, 0, 1],
					[1, 0, 0],
					[0, 0, 0],
					[0, 1, 1],
					[0, 1, 0],
					[0, 0, 0]])
	lv = np.array([0,0,0,1,1,1])
	compute_mi(fm, lv)

