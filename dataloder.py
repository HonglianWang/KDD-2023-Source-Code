#coding=utf-8
'''
load data from a text file, which each line is
node_a node_b
'''
import numpy as np
def read_graph(fp):
    edgelist = []
    with open(fp) as reader:
        for line in reader:
            [node1, node2] = line.strip().split()
            edgelist.append([int(node1), int(node2)])

    unique_nodes = set(np.array(edgelist).reshape(-1,1).squeeze()) # this can be different each time you run it.
    num_nodes = len(unique_nodes)
    return num_nodes, edgelist


def save_indexed_edgelist(fp_old, fp_new): # run this function for only one time!

    # read and re-index the edgelist from original file
    edgelist = []
    with open(fp_old) as reader:
        for line in reader:
            linesplit = line.strip().split()
            # [node1, node2] = line.strip().split() # for dophin dataset
            [node1, node2] = linesplit[1:3] # for sythetic dataset
            edgelist.append([node1, node2])
    reader.close()

    unique_nodes = set(np.array(edgelist).reshape(-1, 1).squeeze())  # this can be different each time you run it.
    num_nodes = len(unique_nodes)
    nodedict = dict(zip(unique_nodes, np.arange(num_nodes)))
    rename_node = lambda edges: [[nodedict[a[0]], nodedict[a[1]]] for a in edges]
    edgelist_new = rename_node(edgelist)

    # save the re-indexed edge-list to new file
    with open(fp_new,'a') as writer:
        for index in range(len(edgelist_new)-1):
            line = str(edgelist_new[index][0]) + ' ' + str(edgelist_new[index][1]) + '\n'
            writer.write(line)
        line = str(edgelist_new[-1][0]) + ' ' + str(edgelist_new[-1][1])
        writer.write(line)
    writer.close()

#
# if __name__ == '__main__':
#     fp_old = 'data/synthetic.txt'
#     fp_new = 'data/synthetic_indexed.txt'
#     save_indexed_edgelist(fp_old, fp_new)
# #