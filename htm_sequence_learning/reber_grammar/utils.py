"""
MODULE DESCRIPTION:
-------------------
Utility function used in generation of Reber Grammar strings

"""

from htm_sequence_learning.reber_grammar.graph_configs import dict_reber_graphs, list_graphs


def reberstring_length(rg_io):
    return len(rg_io[0])


def get_graph_allPossibleTransitions(graph):
    """

    Parameters
    ----------
    graph: a choice of possible reber grammar generators from

    Returns
    -------
    list of all possible trigrams in the reber grammar

    """

    all_possibleTransitions = []
    for i, subsequent_1 in enumerate(graph[0][1]):
        transition_to_1 = graph[0][0][i]

        for j, subsequent_2 in enumerate(graph[transition_to_1][1]):
            all_possibleTransitions.append('A' + subsequent_1 + subsequent_2)

    for startNode in graph:
        if startNode[1] == 'Z':
            continue

        for i, antecedent in enumerate(startNode[1]):
            if len(startNode[1]) == 1:
                transition_to_1 = startNode[0]
            else:
                transition_to_1 = startNode[0][i]

            for j, subsequent_1 in enumerate(graph[transition_to_1][1]):

                if graph[transition_to_1][1] == 'Z':
                    all_possibleTransitions.append(antecedent + subsequent_1)
                    break
                elif len(graph[transition_to_1][1]) == 1:
                    transition_to_2 = graph[transition_to_1][0]
                else:
                    transition_to_2 = graph[transition_to_1][0][j]

                for subsequent_2 in graph[transition_to_2][1]:
                    all_possibleTransitions.append(antecedent + subsequent_1 + subsequent_2)
    return all_possibleTransitions


def count_allTransitions(graph_idx=None, in_reber_strings=None):
    dict_count_allTransitions = {}
    graph = dict_reber_graphs[list_graphs[graph_idx-1]]
    all_possible_transitions = get_graph_allPossibleTransitions(graph=graph)
    for transition in all_possible_transitions:
        counter = 0
        end_at = len(transition) - 1
        for string in in_reber_strings:
            for i in range(len(string) - end_at):
                trans = ''
                for j in range(len(transition)):
                    trans += string[i + j]

                if trans == transition:
                    counter += 1
        dict_count_allTransitions[transition] = counter
    return dict_count_allTransitions


def get_graph_from_dataset(dataset_name):
    graph_idx = dataset_name.split('_')[0][-1]
    return int(graph_idx)
