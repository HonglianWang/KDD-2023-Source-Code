import argparse

def config_initialization():
    parser = argparse.ArgumentParser(description='Run the entire/partial pipeline of FairRandomWalks experiments.')
    parser.add_argument('-proc', type=str, default='radius',
                        help='Part of pipeline to execute: diameter to compute the bubble diameter of all partitions, addition to compute new bubble diameter')
    # parser.add_argument('-algo', type=str, default='baseline', help='If -proc is addition, choose the algo for addition')
    parser.add_argument('-topic', type=str, help='Graph to analyze')
    parser.add_argument('-t', type=int, default=15, help='Length of walks')
    parser.add_argument('-b', type=int, default=2, help='Threshold to define good nodes')
    parser.add_argument('-topk', type=int, default=10, help='Percentage of top-k central nodes to consider')
    parser.add_argument('-maxedges', type=int, default=50, help='Maximum number of edges to add to the graph')
    parser.add_argument('-iter', type=int, default=10, help='Number of iterations')
    parser.add_argument('-unweighted', type=str, default='false', help='Weighted or unweighted graph')
    parser.add_argument('-epsilon', type=float, default=0.1, help='bicriteria epsilon for greedy algoritm, kdd')

    # parser.add_argument('-topic', type=str, help='Graph to analyze')
    args = parser.parse_args()

    return args