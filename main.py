import networkx as nx
import numpy as np
from config import config_initialization
from scipy.sparse import dia_matrix, diags
from KDD_algo import compute_exact_HT_rtoB, Greedy_plusplus, Greedy_plus, Greedy_plusplus_BestofMany
import os, time, pickle

import os, time
import sys
import math
import argparse
import itertools

from collections import defaultdict

import pickle
import multiprocessing as mp
from scipy.sparse import dia_matrix, diags

import parallel_walks
import parallel_addition
import parallel_centrality
from algo import Algorithms
from load_data import LoadData
from random_walks import chunk_random_walk
from utils import load_bubble_diameter, get_bad_and_good_nodes, get_centralities, HittingTime_sample
from matplotlib import pyplot as plt
from controversy_for_edges import compute_ROV_candidate_edge

def KDD_visualization_exactgreedy(A_republink, add_size):
    '''
    get results of Kdd greedy algorithm
    :param A_republink: adjcency matrix of republink paper
    :return: hitting time array [ht after adding k1 edges, ht after adding k2 edges, ..]
    '''
    matrixA = A_republink.A
    G = nx.from_numpy_matrix(matrixA, create_using=nx.Graph)
    R = [i for i in range(numOfNode) if color_nodes[i] == 'red']
    B = [i for i in range(numOfNode) if color_nodes[i] == 'blue']

    K = add_size
    step = int(K/10)
    NumEdges = np.arange(0, K+step, step)  # x values to show in the figure
    x_axis, avgHT, maxHT = [], [], []

    G_new_list = Greedy_plus(G, R, B, NumEdges)


    for i, num in enumerate(NumEdges):
        G_i = G_new_list[i]
        HT = compute_exact_HT_rtoB(G_i, R, B)
        avght = sum(HT) / len(R)
        maxht = max(HT)
        avgHT.append(avght)
        maxHT.append(maxht)
        x_axis.append(num)
    return x_axis, avgHT, maxHT



def calculate_raduis():

    print("Number of walks per node: ", r)


    if not os.path.exists(args.topic + '/'):
        os.makedirs(args.topic + '/')

    # Compute bubble diameter
    result_chunks = parallel_walks.parallelization(A, labels, args.t, color_nodes, r)
    with open(args.topic + '/bubble_diameters.pickle', 'wb') as f:
        pickle.dump(result_chunks, f)

    bubble_diameter = load_bubble_diameter(args.topic, id_matrix_node, args.t)  # {5721464：5.933110367892977}
    red_bubble_diameter, blue_bubble_diameter, bad_red_vertices, bad_blue_vertices = get_bad_and_good_nodes(
        bubble_diameter, id_color, args.b, args.t)  # dictionary {5721464：5.933110367892977}


    # Compute bad nodes centralities Blue
    nodes = np.array([node_id_matrix[n] for n in bad_blue_vertices])  # 0 1 2 3 4
    print('Number blue bad nodes: ', len(nodes))
    result_chunks = parallel_centrality.parallelization(A, labels, args.t, color_nodes, r, nodes)
    with open(args.topic + '/' + 'blue' + '_centralities.pickle', 'wb') as f:
        pickle.dump(result_chunks, f)


    # Compute bad nodes centralities Red
    nodes = np.array([node_id_matrix[n] for n in bad_red_vertices])  # 0 1 2 3 4
    print('Number red bad nodes: ', len(nodes))
    result_chunks = parallel_centrality.parallelization(A, labels, args.t, color_nodes, r, nodes)
    with open(args.topic + '/' + 'red' + '_centralities.pickle', 'wb') as f:
        pickle.dump(result_chunks, f)

    return red_bubble_diameter, blue_bubble_diameter, bad_red_vertices, bad_blue_vertices

def save_statisitcs():
    statisitcs = {}
    statisitcs['num of node'] = len(G.nodes)
    statisitcs['num of Red node'] = len(bad_red_vertices)  # should flip red and blue
    statisitcs['num of blue node'] = len(bad_blue_vertices)
    crossing = 0
    for i in bad_red_vertices.keys():
        for j in bad_blue_vertices.keys():
            if (i,j) in G.edges or (j,i) in G.edges():
                crossing += 1
    statisitcs['RxB'] = crossing
    statisitcs['num of all edges'] = len(G.edges)

    return statisitcs

if __name__ == '__main__':
    topics = ['guns','math_tech','tech_mil','abortion','sociology']
    maxedges = [130, 160, 20, 200, 600]  # as close to number of bad nodes as possible. In this work bad = red


    for idx, topic in enumerate(topics):
        args = config_initialization()
        args.topic = topic
        args.maxedges = maxedges[idx]

        data = LoadData(args.topic, uniform=True)

        id_color = data.id_color
        G = data.G
        node_id_matrix = {n: i for i, n in enumerate(list(G.nodes()))}  # {5721464：0} str id - int id
        id_matrix_node = {j: i for i, j in node_id_matrix.items()}
        labels = np.array([j for i, j in sorted([(i, j) for i, j in id_matrix_node.items()], key=lambda x: x[
            0])])
        color_nodes = np.array([id_color[j] for i, j in sorted([(i, j) for i, j in id_matrix_node.items()],
                                                               key=lambda x: x[
                                                                   0])])  # [red, blue] for int id [0 1 2 3 4 5]

        numOfNode = len(G.nodes)

        Adjacency = nx.adjacency_matrix(G)
        d = diags(1/Adjacency.sum(axis=1).A.ravel())
        A = Adjacency.T.dot(d).T

        delta = 0.05
        eps = 1
        r = int((args.t ** 2) / (eps ** 2) * np.log(1 / delta))
        ITER = 1


        # calculate bubble radius for all nodes, red nodes and blue nodes, save separately.
        red_bubble_diameter, blue_bubble_diameter, bad_red_vertices, bad_blue_vertices = calculate_raduis()  # comment if already computed
        bubble_diameter = load_bubble_diameter(args.topic, id_matrix_node, args.t)  # {5721464：5.933110367892977}
        red_bubble_diameter, blue_bubble_diameter, bad_red_vertices, bad_blue_vertices = get_bad_and_good_nodes(
            bubble_diameter, id_color, args.b, args.t)

        statisitcs = save_statisitcs()
        fp = topic + '/statistics.pickle'
        with open(fp, 'wb') as f:
            pickle.dump(statisitcs, f)

        print ('------------------------')
        print (statisitcs)
        print('-------------------------')

        # calculate candidate edges for ROV algorithm, save to pickle

        ROV_edges = compute_ROV_candidate_edge(data, G, A, node_id_matrix, id_matrix_node, args)
        with open(args.topic + '/rov_candidate_edges.pickle', 'wb') as f:
            pickle.dump(ROV_edges, f)
        #
        with open(args.topic + '/rov_candidate_edges.pickle', 'rb') as f:
            ROV_edges = pickle.load( f)


        K = len(ROV_edges)
        print('number of edges to add to graph is', K)

        nodes = np.array([node_id_matrix[n] for n in bad_red_vertices])
        alg = Algorithms(bad_red_vertices, blue_bubble_diameter, labels, args.t, color_nodes, r, nodes, node_id_matrix, list(G.edges()), ITER)

        baseline_edge = alg._compute_candidate_edge(algorithm='baseline',top_k=100,centrality=[])


        centrality = get_centralities(id_matrix_node, args.topic, 'red', args.t, old=False)  # {str id, centrality}
        top_k = int(len(centrality) / 100 * args.topk)
        RC_candidate_edges = alg._compute_candidate_edge('rand_central', top_k, centrality)


        WRC_candidate_edges = alg._compute_candidate_edge('w_rand_central', top_k, centrality, G)


        REpublik_candidate_edges = alg._compute_candidate_edge('w_pen_central', top_k, centrality, G)
        REpublik_candidate_edges = REpublik_candidate_edges[:K]

        alg_edges = [baseline_edge, ROV_edges, RC_candidate_edges, WRC_candidate_edges, REpublik_candidate_edges]
        algorithms = ['baseline','ROV','RC','WRC','Reppublik','KDD']


        # visualization
        AVG = []
        MAX = []
        n = len(color_nodes)
        Red_int = [i for i in range(n) if color_nodes[i] == 'red']
        Blue_int = [i for i in range(n) if color_nodes[i] == 'blue']

        for i, algorithm in enumerate(algorithms[:-1]):
            edges = alg_edges[i]
            print (algorithm, 'len(edges)',len(edges))
            x_axis, avgHT, maxHT = HittingTime_sample(Adjacency, edges, Red_int, Blue_int, color_nodes, algorithm, K)
            AVG.append(avgHT)
            MAX.append(maxHT)

        x_axis, avgHT, maxHT = KDD_visualization_exactgreedy(Adjacency, K)

        AVG.append(avgHT)
        MAX.append(maxHT)

        fp = topic + '/' + str(time.localtime().tm_hour) + '-' + str(time.localtime().tm_min) + '-AVG.pickle'
        with open(fp, 'wb') as f:
            pickle.dump(AVG,f)

        fp = topic + '/' + str(time.localtime().tm_hour) + '-' + str(time.localtime().tm_min) + '-MAX.pickle'
        with open(fp, 'wb') as f:
            pickle.dump(MAX,f)


        colormap = ['g', 'b', 'r', 'y', 'm', 'c']
        markermap = ['o', 'v', '^', '<', '>', 'x']

        # visualize average hitting time
        fig, ax = plt.subplots()

        for i in range(len(algorithms)):
            ax.plot(x_axis, AVG[i], color=colormap[i], marker=markermap[i], label=algorithms[i])

        plt.xlabel("k")
        plt.ylabel("objective value")
        legend = ax.legend(loc='upper center', shadow=False)

        plt.title(args.topic + ' avg hitting time')

        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig(args.topic + ' avg hitting time')


        # visualize max hitting time
        fig, ax = plt.subplots()
        for i in range(len(algorithms)):
            ax.plot(x_axis, MAX[i], color=colormap[i], marker=markermap[i], label=algorithms[i])

        plt.xlabel("k")
        plt.ylabel("objective value")
        legend = ax.legend(loc='upper center', shadow=False)
        plt.title(args.topic + ' max hitting time')

        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig(args.topic + ' max hitting time')
