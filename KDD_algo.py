import networkx as nx
import random
import numpy as np
import math
import warnings
import heapq
from dataloder import *
from scipy.optimize import linprog


def random_Walk(g):
    walkrecord = []  # list of lists, each list is a random walk sequence, the first element of each list is the starting point of a random walk.

    nodes = list(g.nodes())
    start = random.choice(nodes)
    walkrecord.append([start])
    neigh = list(g.neighbors(start))
    picked_nodes = 1
    walk_length = 0  # length of each random walk

    while True:  # number of sampled nodes to start random walk
        if (len(neigh) == 0) or (walk_length >= 500):  # choose a new node as starting point
            start = random.choice(nodes)
            picked_nodes += 1
            if picked_nodes > 500:
                break
            walkrecord.append([start])
            neigh = list(g.neighbors(start))
            walk_length = 0
        else:  # perfrom random walk as append the visited node to the list.
            next = random.choice(neigh)
            walkrecord[-1].append(next)
            walk_length += 1
            neigh = list(g.neighbors(next))

    return walkrecord


def update_hitting_time(g, walkrecord):
    num_nodes = g.number_of_nodes()
    H = np.zeros([num_nodes, num_nodes])  # hitting time matrix
    update_cnt = np.zeros([num_nodes, num_nodes])  # record how many times is  H(i,j) updated

    for sequence in walkrecord:  # sequence is one random walk sequence
        while len(sequence) > 1:
            start = sequence[0]
            visited = set()
            for i in range(1, len(sequence)):
                node = sequence[i]
                if not node == start:
                    # ----------------------------------------------------------------
                    if not node in visited:
                        H[start, node] += i  # only update H[start, node]
                        update_cnt[start, node] += 1  # if node is seen
                        visited.add(node)  # the first time
                    else:  # from start node
                        pass

                    # ------------------------------------------------------------------
                else:  # if the random walk is [start,...,start,...]
                    pass  # you encounter start again, then break the loop
            sequence = sequence[1:]  # and take the next node after start as a new start node.
            # ------------------------------------------------------------------------------------------------
    return H / update_cnt  # element wise division


def compute_exact_HT_rtoB(G, R, B):
    # Input: networkx udirected graph g (with nodes = 0,1,2,...), a set of Red+Blue nodes that form a partition of nodes in g.
    # Output: A dict with keys = red nodes, and as values the exact hitting times from the red node to hit Blue.
    n = len(G.nodes())

    # Solving a linear program with h_bB = 0 for blue vertices b, and equations d_r* h_rB = d_r + sum_{neighbors i} h_iB.
    c = np.ones(n)
    b = np.zeros(n)
    b[R] = [G.degree(u) for u in R]
    L = nx.laplacian_matrix(G, nodelist=range(n))
    for u in B:
        for v in G.neighbors(u):
            L[u, v] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = linprog(c, A_eq=L, b_eq=b)

    return res.x


def compute_exact_gains(G, R, B, r_bngbs, r):
    # This function computes the gain function when adding the edge (r,b) to G, where (r,b) is some available edge!
    b = r_bngbs[r][0]
    sum1 = sum(compute_exact_HT_rtoB(G, R, B)[R])  # sum HT from r to B, old graph G
    G.add_edge(r, b)
    sum2 = sum(compute_exact_HT_rtoB(G, R, B)[R])  # sum HT from r to B, new candidate graph G+e
    G.remove_edge(r, b)
    return sum1 - sum2


def compute_approximate_gains(G, R, B, r_bngbs, r, epsi, lam, d_R):
    # This function computes the gain function when adding the edge (r,b) to G, where (r,b) is some available edge!
    # print('testing r in R for base edge')
    b = r_bngbs[r][0]
    G.add_edge(r, b)
    gain = estimate_p2(G, R, B, epsi, lam, d_R)
    G.remove_edge(r, b)
    return gain


def LazyGreedy_exact(G, R, B, k):
    # Input: netxorkx g, Red and Blue nodes that form partition of V, k edges to add.
    # Output: new networkx graph with k new edges.
    # Greedy submodular maximization algorithm with Lazy evaluation, by using EXACT hitting times.
    G2 = G.copy()
    # The candidates are red nodes with less than |B| blue neighbors (otherwise we can't add an edge incident to this red node)
    r_bngbs = {r: [b for b in B if b not in G[r]] for r in R}
    # Removing red vertices that are shortcutted to all blue vertices.
    r_bngbs = {r: r_bngbs[r] for r in r_bngbs if len(r_bngbs[r]) > 0}

    # Initialization for the Lazy Greedy evaluation
    gains = [(-compute_exact_gains(G, R, B, r_bngbs, r), r) for r in r_bngbs]
    # Making a min-heap from it (we actually need a max-heap! but we multiply with -1 in the line above)
    heapq.heapify(gains)

    # The first edge to add will simply be the one with highest gain (no lazy eval.)
    r = gains[0][1]
    G2.add_edge(r, r_bngbs[r][0])
    del r_bngbs[r][0]  # deleting the free blue neighbor
    if len(r_bngbs[r]) == 0:  # vertex r is shortcutted to all available blue nodes.
        del r_bngbs[r]
        del gains[0]

    for i in range(k - 1):
        print(i)
        if len(r_bngbs) == 0:
            print('Warning: no more shortcut edges available to add!')
        Lazy = False
        while Lazy is False:
            r = gains[0][1]
            heapq.heapreplace(gains, (-compute_exact_gains(G2, R, B, r_bngbs, r), r))
            if gains[0][1] == r:  # This is the lazy property!
                Lazy = True
                G2.add_edge(r, r_bngbs[r][0])
                del r_bngbs[r][0]
                if len(r_bngbs[r]) == 0:  # vertex r is shortcutted to all available blue nodes.
                    del r_bngbs[r]
                    del gains[0]
    return G2


def Greedy_plus(G, R, B, kvector):
    # Input: netxorkx g, Red and Blue nodes that form partition of V, k edges to add.
    # Output: new networkx graph with k new edges.
    # Greedy+ algorithm from the paper.

    # kvector is an INCREASING vector where we want the output recorded (e.g. k = [1, 50, 100, 150])
    kmax = kvector[-1:][0]
    G2 = G.copy()
    listofGraphs = []

    epsi = 0.1
    lam = 0.1

    # The candidates are red nodes with less than |B| blue neighbors (otherwise we can't add an edge incident to this red node)
    r_bngbs = {r: [b for b in B if b not in G[r]] for r in R}
    # Removing red vertices that are shortcutted to all blue vertices.
    r_bngbs = {r: r_bngbs[r] for r in r_bngbs if len(r_bngbs[r]) > 0}
    # The average degree of the red nodes
    d_R = np.average([val for (node, val) in G.degree()])

    # Take a subset X of the red nodes for our estimate
    X = random.sample(R, 20)

    # Initialization for the Greedy evaluation
    for i in range(kmax):
        print(i)
        if i in kvector:
            listofGraphs.append(G2.copy())
        gains = [(compute_approximate_gains(G2, X, B, r_bngbs, r, epsi, lam, d_R), r) for r in r_bngbs]
        r = min(gains)[1]
        G2.add_edge(r, r_bngbs[r][0])
        del r_bngbs[r][0]
        if len(r_bngbs[r]) == 0:  # vertex r is shortcutted to all available blue nodes.
            del r_bngbs[r]

    listofGraphs.append(G2.copy())
    return listofGraphs


def Greedy_plusplus(G, R, B, kvector):
    # Input: netxorkx g, Red and Blue nodes that form partition of V, k edges to add.
    # Output: new networkx graph with k new edges.
    # Greedy+ algorithm from the paper with LAZY evaluation

    # kvector is an INCREASING vector where we want the output recorded (e.g. k = [1, 50, 100, 150])
    kmax = kvector[-1:][0]
    G2 = G.copy()
    listofGraphs = []

    epsi = 0.1
    lam = 0.1

    # The candidates are red nodes with less than |B| blue neighbors (otherwise we can't add an edge incident to this red node)
    r_bngbs = {r: [b for b in B if b not in G[r]] for r in R}
    # Removing red vertices that are shortcutted to all blue vertices.
    r_bngbs = {r: r_bngbs[r] for r in r_bngbs if len(r_bngbs[r]) > 0}
    # The average degree of the red nodes
    d_R = np.average([val for (node, val) in G.degree()])

    # Take a subset X of the red nodes for our estimate
    X = random.sample(R, math.ceil(len(R) / 5))

    # Initialization for the Lazy Greedy evaluation
    gains = [(compute_approximate_gains(G, X, B, r_bngbs, r, epsi, lam, d_R), r) for r in r_bngbs]
    # Making a min-heap from it (we actually need a max-heap! but we multiply with -1 in the line above)
    heapq.heapify(gains)

    if kvector[0] == 0:
        listofGraphs.append(G2.copy())

    # The first edge to add will simply be the one with highest gain (no lazy eval.)
    r = gains[0][1]
    G2.add_edge(r, r_bngbs[r][0])
    del r_bngbs[r][0]  # deleting the free blue neighbor
    if len(r_bngbs[r]) == 0:  # vertex r is shortcutted to all available blue nodes.
        del r_bngbs[r]
        del gains[0]

    for i in range(kmax - 1):
        print(i)
        if i + 1 in kvector:
            listofGraphs.append(G2.copy())
        if len(r_bngbs) == 0:
            print('Warning: no more shortcut edges available to add!')
        Lazy = False
        while Lazy is False:
            r = gains[0][1]
            heapq.heapreplace(gains, (compute_approximate_gains(G2, X, B, r_bngbs, r, epsi, lam, d_R), r))
            if gains[0][1] == r:  # This is the lazy property!
                Lazy = True
                G2.add_edge(r, r_bngbs[r][0])
                del r_bngbs[r][0]
                if len(r_bngbs[r]) == 0:  # vertex r is shortcutted to all available blue nodes.
                    del r_bngbs[r]
                    del gains[0]
    # Appending the last kmax
    listofGraphs.append(G2.copy())
    return listofGraphs


def Greedy_plusplus_BestofMany(G, R, B, kvector, iterations):
    current = Greedy_plusplus(G, R, B, kvector)
    current_value = np.average(compute_exact_HT_rtoB(current[-1:][0], R, B))
    for i in range(iterations - 1):
        test = Greedy_plusplus(G, R, B, kvector)
        test_value = np.average(compute_exact_HT_rtoB(test[-1:][0], R, B))

        if test_value <= current_value:
            current = test.copy()
            current_value = np.average(compute_exact_HT_rtoB(current[-1:][0], R, B))

    return current


def estimate_p2(G, R, B, epsi, lam, d_R):
    # This estimates the average hitting time (over all red nodes) from R to B.
    # We simulate a walk from every red node.
    # Epsi and lam are parameters determining how long the walks will run (see paper)
    # d_R is the average degree of the red nodes in G.
    # DEFAULT settings: epsi = lam = 0.2
    estimate = np.zeros(len(R))

    l = 10
    t = 10
    # n = len(G.nodes())
    # l = math.ceil(math.log(d_R/epsi/(1-lam))/math.log(1/lam))
    # t = math.ceil(((l/epsi)**2)*math.log10(n))

    i = 0
    for r in R:
        estimate[i] = run_rw(G, R, B, r, l, t)
        i = i + 1

    return np.average(estimate)


def run_rw(G, R, B, r, l, t):
    # Run a r.w. from node r (in R), with max. length at most l or until absorbed by B.
    # Record the empirical average length over t repeats.
    estimates = np.zeros(t)
    for i in range(t):
        steps = 0
        current = r
        while steps < l:
            if current in B:
                estimates[i] = steps
                break
            else:
                current = random.choice(list(G[current]))
                steps = steps + 1
                if steps >= l:
                    estimates[i] = steps
                    break
    return np.average(estimates)

#
# # Main
# fp = 'data/dophin.txt'
# num_nodes, edge_list = read_graph(fp)
# G0 = nx.Graph()
# G0.add_edges_from(edge_list)
#
# # Extract largest connected component.
# Gcc = sorted(nx.connected_components(G0), key=len, reverse=True)
# G1 = G0.subgraph(Gcc[0])
#
# # Relabel the nodes such that they start from 0,1,...
# A = nx.to_scipy_sparse_matrix(G1)
# G = nx.from_scipy_sparse_matrix(A, parallel_edges=False)
#
# # walkrecord = random_Walk(g)
# # H = update_hitting_time(g,walkrecord)
# # print (H)
#
# # Test compute exact HT
# B = [12, 20, 4, 5, 6, 7, 8, 10]
# R = [v for v in G.nodes() if v not in B]
# # print(compute_exact_HT_rtoB(G, R, B))