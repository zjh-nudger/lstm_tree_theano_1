import numpy as np
from nltk import Tree


def reverse_toposort(tree):
    """Reverse topological sorting

    Always go from the leaves first and then build up
    from the bottom up.
    """
    btree = binarize_tree(tree)
    num_leaves = tag_leaves(btree)
    ordering_list = [(i, 0, 0) for i in range(num_leaves)]
    num_nodes = recurs_reverse_toposort(btree, num_leaves, ordering_list)
    assert(num_nodes == (2 * num_leaves - 1))
    return np.array(ordering_list, dtype='int64'), num_leaves


def find_parse_tree(relation, arg_pos):
    assert arg_pos == 1 or arg_pos == 2
    arg_token_addresses = relation.arg_token_addresses(arg_pos)
    if arg_pos == 1:
        arg_token_addresses = _truncate_to_last_sentence(arg_token_addresses)
    elif arg_pos == 2:
        arg_token_addresses = _truncate_to_first_sentence(arg_token_addresses)
    sentence_index = arg_token_addresses[0][3]
    parse_tree_string = relation.parse[relation.doc_id]['sentences'][sentence_index]['parsetree']
    parse_tree = Tree(parse_tree_string)[0]
    return parse_tree
    #first_token = arg_token_addresses[0][4]
    #last_token = arg_token_addresses[-1][4]
    #tp = parse_tree.treeposition_spanning_leaves(first_token, last_token)
    #print parse_tree[tp]
    #return parse_tree[tp]

def left_branching_tree(relation, arg_pos):
    assert arg_pos == 1 or arg_pos == 2
    arg_tokens = relation.arg_tokens(arg_pos)
    leaves = [Tree('(-1 %s)' % t) for t in arg_tokens]
    root = None
    for i, leaf in enumerate(leaves):
        if i == 0:
            root = leaf
        else:
            root = Tree(-1, [root, leaf])
    return root

def tag_leaves(t):
    """Tag the leaf nodes with indices in the linear order

    """
    def recurs_tag_leaves(t, num_leaves_so_far):
        # leaf. should have updated here. But it's not an object
        if not isinstance(t, Tree):
            return num_leaves_so_far + 1

        for i, child in enumerate(t):
            num_leaves_so_far = recurs_tag_leaves(child, num_leaves_so_far)
            # awkward because the leaf node is not an object... it's an int
            if not isinstance(child, Tree):
                t[i] = num_leaves_so_far - 1
        return num_leaves_so_far
    return recurs_tag_leaves(t, 0)


def recurs_reverse_toposort(t, num_nodes, ordering_list):
    if not isinstance(t, Tree):
        # ordering_list.append((t, 0, 0))
        return num_nodes

    assert(len(t) == 2)
    child_indices = []
    for child in t:
        num_nodes = recurs_reverse_toposort(child, num_nodes, ordering_list)
        if isinstance(child, Tree):
            child_indices.append(child.node)
        else:
            child_indices.append(child)
    t.node = num_nodes
    num_nodes += 1
    ordering_list.append((t.node, child_indices[0], child_indices[1]))
    return num_nodes


def binarize_tree(t):
    """Convert all n-nary nodes into left-branching subtrees

    Returns a new tree. The original tree is intact.
    """
    def recurs_binarize_tree(t):
        if t.height() <= 2:
            return t[0]

        if len(t) == 1:
            return recurs_binarize_tree(t[0])
        elif len(t) == 2:
            new_children = []
            for i, child in enumerate(t):
                new_children.append(recurs_binarize_tree(child))
            return Tree(-1, new_children)
        else:
            left_child = recurs_binarize_tree(Tree(-1, t[0:-1]))
            right_child = recurs_binarize_tree(t[-1])
            return Tree(-1, [left_child, right_child])
    return recurs_binarize_tree(t)


def _truncate_to_last_sentence(token_list):
    sentence_indices = [t[3] for t in token_list]
    last_sentence_index = max(sentence_indices)
    return [t for t in token_list if t[3] == last_sentence_index]


def _truncate_to_first_sentence(token_list):
    sentence_indices = [t[3] for t in token_list]
    first_sentence_index = min(sentence_indices)
    return [t for t in token_list if t[3] == first_sentence_index]
