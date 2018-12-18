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
	RW_x = np.loadtxt('plots/lesmis_walk_wa/trainingIteration_3200_expectedGraph.txt'.format(graph_name))
	edge_x = np.loadtxt('plots/lesmis_edge_wa/trainingIteration_3200_expectedGraph.txt'.format(graph_name))
	#RW_c = np.loadtxt('plots/{}_rw_expected_correct.txt'.format(graph_name))

	G = nx.read_gml('../data/{}.gml'.format(graph_name))
	_A_obs = nx.adjacency_matrix(G)
	A = _A_obs.todense()
	N = A.shape[0]



	L = nx.normalized_laplacian_matrix(G).todense()
	eig_vals , eig_vecs = linalg. eig ( L )
	eig_list = zip( eig_vals , np . transpose ( eig_vecs ) )
	eig_list . sort ( key = lambda x : x [0])
	u = np.asarray([ u_i . real for u_i in eig_list [ -2][1]])[0][0]

	x,y = expected_against_fiedler(RW_x,u,N)
	fig, ax = plt.subplots()
	plt.scatter(x,y,color='r',s=10,label='Degree Random Walk From Uniform')

	#x_c,y = expected_against_fiedler(RW_c,u,N)
	#plt.scatter(x_c,y,color='m',s=10,label='Degree Random Walk From Stationary')

	x_f,y = expected_against_fiedler(edge_x,u,N)
	plt.scatter(x_f,y,color='b',s=10,label='Fastest Mixing Random Walk')
	ax.legend()
	plt.savefig('plots/{}_expected_fiedler.pdf'.format(graph_name))
	plt.gcf().clear()
	stats = {}
	stats['Mean Expected Edge DRW From Uniform']=np.mean(x)
	stats['Std Expected Edge DRW From Uniform']=np.mean(x)
	#stats['Mean Expected Edge DRW From Stationary']=np.mean(x_c)
	#stats['Std Expected Edge DRW From Stationary']=np.mean(x_c)
	stats['Mean Expected Edge edge']=np.mean(x_f)
	stats['Std Expected Edge edge']=np.mean(x_f)



	rw_entropy = utils.graph_entropy_matrix(RW_x)
	#rw_entropy_c = np.loadtxt('plots/{}_rw_entropy_correct.txt'.format(graph_name)).flatten()
	edge_entropy = utils.graph_entropy_matrix(edge_x)

	#print(max(rw_entropy))
	#print(max(edge_entropy))

	print(np.mean(rw_entropy))
	print(np.std(rw_entropy))

	print(np.mean(edge_entropy))
	print(np.std(edge_entropy))
	stats['Mean Expected Edge Entropy DRW From Uniform']=np.mean(rw_entropy)
	stats['Std Expected Edge Entropy DRW From Uniform']=np.mean(rw_entropy)
	#stats['Mean Expected Edge Entropy DRW From Stationary']=np.mean(rw_entropy_c)
	#stats['Std Expected Edge Entropy DRW From Stationary']=np.mean(rw_entropy_c)
	stats['Mean Expected Edge Entropy edge']=np.mean(edge_entropy)
	stats['Std Expected Edge Entropy edge']=np.mean(edge_entropy)
	stats['Mean Expected Edge Entropy edge Over Non-zeros']=np.mean([x for x in edge_entropy if x != 0])
	stats['Mean Expected Edge Entropy DRW Over Non-zeros']=np.mean([x for x in rw_entropy if x != 0])
	stats['Std Expected Edge Entropy edge Over Non-zeros']=np.std([x for x in edge_entropy if x != 0])
	stats['Std Expected Edge Entropy DRW Over Non-zeros']=np.std([x for x in rw_entropy if x != 0])
	stats['Total Expected Edge Entropy edge']=np.sum(edge_entropy)
	stats['Total Expected Edge Entropy DRW']=np.sum(rw_entropy)
	#plt.hist2d(x, y, bins=(50,50), cmap=plt.cm.Reds)
	#plt.show()
	bins = np.arange(0,max([max(edge_entropy),max(rw_entropy)])+.005,.005)
	plt.hist([x for x in rw_entropy if x != 0], bins, alpha=0.5, label='drw uniform')
	plt.hist([x for x in rw_entropy_c if x != 0], bins, alpha=0.5, label='drw stationary')
	plt.hist([x for x in edge_entropy if x != 0], bins, alpha=0.5, label='edge')
	plt.legend(loc='upper right')
	plt.savefig('plots/{}_entropy_dist_nz.pdf'.format(graph_name))
	plt.gcf().clear()


	plt.hist(rw_entropy, bins, alpha=0.5, label='drw uniform')
	#plt.hist(rw_entropy_c, bins, alpha=0.5, label='drw stationary')
	plt.hist(edge_entropy, bins, alpha=0.5, label='edge')
	plt.legend(loc='upper right')
	plt.savefig('plots/{}_entropy_dist.pdf'.format(graph_name))
	plt.gcf().clear()

	f = open('plots/{}_distn_stats.txt'.format(graph_name),"w")
	f.write( str(stats) )
	f.close()



