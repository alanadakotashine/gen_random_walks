import sys
import numpy as np
import networkx as nx
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import seaborn as sns
import utils

def expected_against_fiedler(X,u,N):
	x = []
	y = []
	for i in range(N):
		for j in range(N):
			x.append(X[i,j])
			y.append(u[i]*u[j])
	return x,y

def gnp(X):
    X = np.triu(X)
    G = np.random.binomial(1,p=X)
    G = G+G.T
    return G

def genExpected_fromWalks(edge_probs,targetSum):
        #edge_probs = gr_edgeProb.data
        #assuming that all edge probs were included 
        #n = gr_edgeProb.shape[0]
        #edge_probs = gr_edgeProb.todense()
        np.fill_diagonal(edge_probs, 0)
        edge_probs = (edge_probs / np.sum(edge_probs))*targetSum
        largerThanOne = len(edge_probs[edge_probs > 1])
        if (len(edge_probs[(edge_probs>0)])<targetSum):
        	edge_probs[(edge_probs>0)]=1
        	return edge_probs
        while largerThanOne > 0:
            edge_probs[edge_probs > 1]=1
            scale = (targetSum/float(edge_probs.sum()))
            edge_probs = edge_probs*(scale)
            largerThanOneEntries = edge_probs[edge_probs > 1]
            largerThanOne = largerThanOneEntries.size
        #gr_edgeProb = sp3.csr_matrix(edge_probs)
        return edge_probs

def gen_graphs(X_from_walks,target):
	X = genExpected_fromWalks(X_from_walks,target.sum())
	print(X)
	sps_truth = utils.sp(target)
	sp_emds = []
	sgl2s = []
	for i in range(20):
		sampled_graph = gnp(X)
		sps_gen = utils.sp(sampled_graph)
		sp_emd = utils.emd(sps_truth,sps_gen)
		sgl2 = (utils.specGap(target)-utils.specGap(sampled_graph))**2
		sp_emds.append(sp_emd)
		sgl2s.append(sgl2)
	return sp_emds, sgl2s







if __name__ == "__main__":
	graph_name = sys.argv[1]
	mix_times = []
	G = nx.read_gml('../data/{}.gml'.format(graph_name))
	_A_obs = nx.adjacency_matrix(G)
	A = _A_obs.todense()
	truth = utils.randWalkMatrix_mix_time(np.array(A))
	print('TRUTH')
	print(truth)
	mix_times_e = []
	mix_times_truth = []
	for i in range(1,7):
		X = np.loadtxt('plots/trainingIteration_{}_expectedGraph.txt'.format(i*500))
		X_e = np.loadtxt('plots/lesmis_edge_wd_em/trainingIteration_{}_expectedGraph.txt'.format(i*500))
		mix_times.append(utils.randWalkMatrix_mix_time(X))
		mix_times_e.append(utils.randWalkMatrix_mix_time(X_e))
		mix_times_truth.append(truth)
	plt.plot(range(len(mix_times)),mix_times,label='Train w/ Path Length 2 No Temp Decay')
	plt.plot(range(len(mix_times_e)),mix_times_e,label='Train w/ Path Length 2')
	plt.plot(range(len(mix_times_truth)),mix_times_truth, label='True Mix-time')
	plt.legend(loc='best')
	plt.savefig('plots/lesmis_mixtimes_tempdecay.pdf')




