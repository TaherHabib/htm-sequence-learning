
dict_reber_graphs = {
    'graph_1': [[(1, 5), ('T', 'P')], [(1, 2), ('S', 'X')],
                [(3, 5), ('V', 'X')], [(6), ('Z')],
                [(4, 2, 3), ('T', 'P', 'S')], [(4), ('V')]],

    'graph_2': [[(1, 5), ('T', 'P')], [(1, 2), ('S', 'P')],
                [(3, 5), ('V', 'X')], [(6), ('Z')],
                [(4, 3), ('T', 'X')], [(1, 4), ('V', 'S')]],  # the one used as a standard (and in the thesis).
                                                              # Doesn't have a uniform distribution of trigrams.
    'graph_3': [[(1, 5), ('T', 'P')], [(1, 2), ('S', 'P')],
                [(3, 5), ('V', 'X')], [(6), ('Z')],
                [(4, 3), ('T', 'X')], [(2, 4), ('V', 'S')]],

    'graph_4': [[(1, 5), ('T', 'P')], [(4, 2), ('X', 'P')],
                [(3, 2), ('V', 'S')], [(6), ('Z')],
                [(5, 3), ('T', 'X')], [(1, 4), ('V', 'S')]],

    'graph_5': [[(1, 5), ('T', 'P')], [(4, 2), ('X', 'P')],
                [(3, 1), ('V', 'S')], [(6), ('Z')],
                [(5, 3), ('T', 'X')], [(2, 4), ('V', 'S')]]  # symmetric one with uniform distribution of trigrams
}

list_graphs = list(dict_reber_graphs.keys())
chars = 'ATSXPVZ'
embedded_chars = 'TP'
