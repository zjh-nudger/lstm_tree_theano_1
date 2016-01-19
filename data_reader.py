import json

class DRelation(object):
	"""Implicit discourse relation object

	The object is created from the CoNLL-json formatted data.
	The format can be a bit clunky to get certain information. 
	So convenient methods should be implemented here mostly to be used
	by the feature functions
	"""

	def __init__(self, relation_dict, parse):
		self.relation_dict = relation_dict
		self.parse = parse	
		self._arg_tokens = {}
		self._arg_tokens[1] = None
		self._arg_tokens[2] = None

		self._arg_words = {}
		self._arg_words[1] = None
		self._arg_words[2] = None

		self._arg_tree = {}
		self._arg_tree[1] = None
		self._arg_tree[2] = None

		self._arg1_tree = None
		self._arg1_tree_token_indices = None
		self._arg2_tree = None
		self._arg2_tree_token_indices = None

	@property
	def senses(self):
		return self.relation_dict['Sense']

	def arg_words(self, arg_pos):
		"""Returns a list of Word objects"""
		assert(arg_pos == 1 or arg_pos == 2)
		if self._arg_words[arg_pos] is None:
			key = 'Arg%s' % arg_pos
			word_list = self.relation_dict[key]['TokenList']	
			self._arg_words[arg_pos] = [Word(x, self.parse[self.doc_id]) for x in word_list]
		return self._arg_words[arg_pos]

	def arg_tree(self, arg_pos):
		"""Extract the tree for the

		Returns:
			1) tree string
			2) token indices (not address tuples) of that tree. 
		"""
		assert(arg_pos == 1 or arg_pos == 2)
		if self._arg_tree[arg_pos] is None:
			trees, sentence_indices = self.arg_trees(arg_pos)	
			if arg_pos == 1:
				tree = trees[-1]
				sentence_index = sentence_indices[-1]
			elif arg_pos == 2:
				tree = trees[0]
				sentence_index = sentence_indices[0]

			key = 'Arg%s' % arg_pos
			token_indices = [x[4] for x in self.relation_dict[key]['TokenList'] if x[3] == sentence_index]
			self._arg_tree[arg_pos] = (tree, token_indices)
		return self._arg_tree[arg_pos]

	def arg_token_addresses(self, arg_pos):
		assert(arg_pos == 1 or arg_pos == 2)
		key = 'Arg%s' % arg_pos
		return self.relation_dict[key]['TokenList']	

	@property
	def doc_id(self):
		return self.relation_dict['DocID']

	@property
	def relation_id(self):
		return self.relation_dict['ID']

	@property
	def doc_relation_id(self):
		return '%s_%s' % (self.doc_id, self.relation_id)

	def arg_tokens(self, arg_pos):
		"""Returns a list of raw tokens"""
		assert(arg_pos == 1 or arg_pos == 2)
		if self._arg_tokens[arg_pos] is None:
			key = 'Arg%s' % arg_pos
			token_list = self.relation_dict[key]['TokenList']	
			self._arg_tokens[arg_pos] = [self.parse[self.doc_id]['sentences'][x[3]]['words'][x[4]][0] for x in token_list]
		return self._arg_tokens[arg_pos]

	def arg_trees(self, arg_pos):
		key = 'Arg%s' % arg_pos
		token_list = self.relation_dict[key]['TokenList']	
		sentence_indices = set([x[3] for x in token_list])
		return [self.parse[self.doc_id]['sentences'][x]['parsetree'] for x in sentence_indices], list(sentence_indices)


	def __repr__(self):
		return self.relation_dict.__repr__()

	def __str__(self):
		return self.relation_dict.__str__()

class Word(object):
	"""Word class wrapper

	[u"'ve",
		{u'CharacterOffsetBegin':2449,
		u'CharacterOffsetEnd':2452,
		u'Linkers':[u'arg2_15006',u'arg1_15008'],
		u'PartOfSpeech':u'VBP'}]
	"""

	def __init__(self, word_address, parse):
		self.word_address = word_address
		self.word_token, self.word_info = parse['sentences'][word_address[3]]['words'][word_address[4]]

	@property
	def pos(self):
		return self.word_info['PartOfSpeech']

	@property
	def lemma(self):
		return self.word_info['Lemma']

	@property
	def sentence_index(self):
		return self.word_address[3]

def extract_implicit_relations(data_folder, label_function=None):
	parse_file = '%s/pdtb-parses-plus.json' % data_folder
	parse = json.load(open(parse_file))

	relation_file = '%s/pdtb-data-plus.json' % data_folder
	relation_dicts = [json.loads(x) for x in open(relation_file)]
	relations = [DRelation(x, parse) for x in relation_dicts if x['Type'] == 'Implicit']
	if label_function is not None:
		relations = [x for x in relations if label_function.label(x) is not None]
	return relations

