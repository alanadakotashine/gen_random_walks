import networkx as nx
import numpy as np
import utils
from scipy import sparse

def gnp(X):
    X = np.triu(X)
    G = np.random.binomial(1,p=X)
    G = G+G.T
    return G

def print_evaluate(A_matrix,sampled_graph):
	stats = utils.compute_graph_statistics(sampled_graph)


	sps_truth = utils.sp(A_matrix)
	sps_gen = utils.sp(sampled_graph)

	stats['sp_emd']= utils.emd(sps_truth,sps_gen)
	s = utils.specGap(A_matrix)
	stats['spec_gap']=(s-utils.specGap(sampled_graph))**2

	print(stats)

G = nx.read_gml('../data/football.gml')
_A_obs = nx.adjacency_matrix(G).todense()



_N = _A_obs.shape[0]


p = _A_obs.sum()/float((_N**2))

S = np.ones((_N,_N))*p 

_graph = utils.graph_from_scores(sparse.csr_matrix(S), _A_obs.sum())

gnp_graph = gnp(S)

print_evaluate(_A_obs,_graph)
print_evaluate(_A_obs,gnp_graph)





