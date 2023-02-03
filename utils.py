import pickle
import random
import itertools
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import warnings
from scipy.optimize import linprog

from KDD_algo import compute_exact_HT_rtoB


def load_bubble_diameter(topic, id_matrix_node, t):
    """Loads the bubble diameter.

    :topic: specify graph
    :id_matrix_node: dictionary (id adj matrix, graph id)
    :t: length of random walks
    """
    # CHANGE PCA
    with open(topic + '/bubble_diameters.pickle', 'rb') as infile:
        data = pickle.load(infile)

    bubble_diameter = {}
    for k in range(len(data)):
        for i, j in zip(list(data[k][0]), list(data[k][1])):
            bubble_diameter[id_matrix_node[i]] = j[0]

    return bubble_diameter


def get_bad_and_good_nodes(bubble_diameter, id_color, b, t, G=None, color=None):
    """Returns dictionaries (node, bubble diameter) of bad and good nodes
    for each of the two partitions.

    :bubble_diameter: dict(node, diameter)
    :id_color: dict(node, color)
    :b: threshold to define god nodes
    :t: length of random walks
    """
    # Get diameter of nodes belonging to different partitions
    red_bubble_diameter = {i: j for i, j in bubble_diameter.items() if id_color[i] == 'red'}
    print('NUMBER of RED NODES: ', len(red_bubble_diameter))
    blue_bubble_diameter = {i: j for i, j in bubble_diameter.items() if id_color[i] == 'blue'}

    # Get bad vertices

    # original code
    # bad_red_vertices = {i:j for i,j in red_bubble_diameter.items() if j >= np.mean(list(red_bubble_diameter.values()))} # t/2
    # print('NUMBER of RED BAD NODES: ', len(bad_red_vertices))
    # bad_blue_vertices = {i:j for i,j in blue_bubble_diameter.items() if j >= np.mean(list(blue_bubble_diameter.values()))} # t/2
    #
    # print('"%" bad red vertices: ', len(bad_red_vertices)/len(red_bubble_diameter)*100)
    #

    # revised code
    bad_red_vertices = {i: j for i, j in red_bubble_diameter.items()}  # t/2
    print('NUMBER of RED BAD NODES: ', len(bad_red_vertices))
    bad_blue_vertices = {i: j for i, j in blue_bubble_diameter.items()}  # t/2
    print('NUMBER of BLUE BAD NODES: ', len(bad_blue_vertices))

    return red_bubble_diameter, blue_bubble_diameter, bad_red_vertices, bad_blue_vertices


def get_centralities(id_matrix_node, topic, color, t, old=True):
    """Returns nodes centralities.

    :id_matrix_node: dict(id adj matrix, node)
    :topic: graph of interes
    :color: partition
    :t: length of random walks
    """

    with open(topic + '/' + color + '_centralities.pickle', 'rb') as infile:
        hitting = pickle.load(infile)

    hitting_time = defaultdict(dict)

    for i in range(len(hitting)):
        index_nodes_bad = hitting[i][0]
        time_hit = hitting[i][1]
        hit_nodes = hitting[i][2]

        for h, n_h in enumerate(hit_nodes):
            for b, n_b in enumerate(index_nodes_bad):
                hitting_time[id_matrix_node[n_h]][id_matrix_node[n_b]] = time_hit[b, h]

    centrality = {i: np.mean([t - l for k, l in j.items()]) for i, j in hitting_time.items()}

    return centrality


def sort_edges(candidate_edges, K):
    """Return sorted edges to add to the graph.

    :candidate_edges:
    :K: list of total nodes to add at each iteration
    """

    new_edges = []
    added_edges = defaultdict(int)
    for i, j, k in candidate_edges:
        added_edges[i] = 0.1

    # for l in range(K[-1]):
    #     print(l)
    #     #cand_start =
    #     candidate = (candidate_edges[0][0], candidate_edges[0][1])
    #     new_edges += [(candidate[0], candidate[1])]
    #     added_edges[candidate[0]] += 1
    #     candidate_edges.remove(candidate_edges[0])
    #     candidate_edges.append(candidate_edges[0])

    for l in range(K):
        # print(l)
        # cand_start =
        candidate = (candidate_edges[0][0], candidate_edges[0][1])
        new_edges += [(candidate[0], candidate[1])]
        added_edges[candidate[0]] += 1
        candidate_edges.remove(candidate_edges[0])
        candidate_edges.append(candidate_edges[0])

    return new_edges


def get_tok_nodes(G, nodes, perc_k=5):
    """ Return top-k% nodes.

    :G: graph
    :nodes: list of nodes to do the top-k%
    :perc_k: value of K
    """

    red_degrees = {}
    for i in nodes:
        red_degrees[i] = G.degree(i)
    k = int(len(red_degrees) / 100 * perc_k)
    red_topk = [i for i, j in sorted(red_degrees.items(), key=lambda x: x[1], reverse=True)][:k]

    return red_topk


def get_candidate_edges(G, red_nodes, blue_nodes, perc_k=5):
    """ Return list of candidate edges.

    :G: graph
    :red_nodes: list of nodes to attach edge
    :blue_nodes: edge endpoints
    :nodes: list of nodes to do the top-k%
    :perc_k: value of K
    """

    red_topk = get_tok_nodes(G, red_nodes, perc_k=5)
    blue_topk = get_tok_nodes(G, blue_nodes, perc_k=5)

    candidate_edges = list(itertools.product(red_topk, blue_topk))
    # Add reverse edges since
    candidate_edges = candidate_edges + [(j, i) for i, j in candidate_edges]

    return candidate_edges


def HittingTime(A_republink, add_edges, Red_int, Blue_int, color_nodes):
    '''
    G:  the directed graph read from click stram
    add_edges: list of tuples, each tuple is (int id, int id)
    R: list of all rea edges
    B: ilst of all blue edges

    Return None, plot a figure of avg hitting time from R to B, and the maximum hitting time from R to B.
    '''

    debug = 'start'
    for (i, j) in add_edges:
        if color_nodes[i] == color_nodes[j]:
            print(color_nodes[i], color_nodes[j])

    step = int(np.floor(len(add_edges) / 10))
    NumEdges = np.arange(0, len(add_edges), step)
    x_axis, avgHT, maxHT = [], [], []

    for i, num in enumerate(NumEdges):
        edges_subset = add_edges[: num]
        x_axis.append(num)

        matrixA = A_republink.A
        G = nx.from_numpy_matrix(matrixA, create_using=nx.Graph)
        G.add_edges_from(edges_subset)

        HT = compute_exact_HT_rtoB(G, Red_int, Blue_int)
        avght = sum(HT) / (1. * len(Red_int))
        maxht = max(HT)

        avgHT.append(avght)
        maxHT.append(maxht)

    return x_axis, avgHT, maxHT


def HittingTime_sample(A_republink, add_edges, Red_int, Blue_int, color_nodes, algorithm, K):
    '''
    G:  the directed graph read from click stram
    add_edges: list of tuples, each tuple is (int id, int id)
    R: list of all rea edges
    B: ilst of all blue edges

    Return None, plot a figure of avg hitting time from R to B, and the maximum hitting time from R to B.
    '''

    debug = 'start'
    for (i, j) in add_edges:
        if color_nodes[i] == color_nodes[j]:
            print(color_nodes[i], color_nodes[j])

    step = int(np.floor(K / 10))
    NumEdges = np.arange(0, K + step, step)
    x_axis, avgHT, maxHT = [], [], []

    if algorithm in ['baseline', 'RC', 'WRC']:
        to_add = []
        for i, num in enumerate(NumEdges):
            print('sampling for the ', i, '-th edges in K')
            to_add = to_add + random.sample(add_edges, num - len(to_add))
            add_edges = list(set(add_edges).difference(set(to_add)))

            x_axis.append(num)

            matrixA = A_republink.A
            G = nx.from_numpy_matrix(matrixA, create_using=nx.Graph)
            G.add_edges_from(to_add)

            HT = compute_exact_HT_rtoB(G, Red_int, Blue_int)
            avght = sum(HT) / (1. * len(Red_int))
            maxht = max(HT)

            avgHT.append(avght)
            maxHT.append(maxht)

    else:
        for i, num in enumerate(NumEdges):
            edges_subset = add_edges[: num]
            x_axis.append(num)

            matrixA = A_republink.A
            G = nx.from_numpy_matrix(matrixA, create_using=nx.Graph)
            G.add_edges_from(edges_subset)

            HT = compute_exact_HT_rtoB(G, Red_int, Blue_int)
            avght = sum(HT) / (1. * len(Red_int))
            maxht = max(HT)

            avgHT.append(avght)
            maxHT.append(maxht)

    return x_axis, avgHT, maxHT