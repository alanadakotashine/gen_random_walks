import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import utils
import numpy.linalg as linalg
import itertools

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

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


def againstFiedler(z,u,A,f,non_zero_only):
	if non_zero_only ==1:
		nz_coords = np.nonzero(z)
		nz_coords = zip(nz_coords[0],nz_coords[1])
	else:
		nz_coords = list(itertools.combinations(range(z.shape[0]),2))
	diffs = []
	l2_s = []
	colors_s = []
	for (v1,v2) in nz_coords:
		diffs.append(z[v1,v2])
	diffs_s = [x for x, _ in sorted(zip(diffs,nz_coords), key=lambda pair: pair[0])]
	coords_s = [x for _, x in sorted(zip(diffs,nz_coords), key=lambda pair: pair[0])]
	k=0
	for (v1,v2) in coords_s:
		l2_s.append(u[v1]*u[v2])
		if u[v1]*u[v2]< -.09:
			if (A[v1,v2]>0.01):
				print('COORDINATE I DID NOT GET')
				print(v1)
				print(v2)
				print(u[v1]*u[v2])
		if z[v1,v2]>0:
			if A[v1,v2]<=.01:
				if f==1:
					print(hi)
		if (A[v1,v2]>0.01):
			colors_s.append('b')
		else:
			colors_s.append('r')
		#if diffs_s[k]>0:
	#		if colors_s[-1]=='r':
	#			print('error')
		k=k+1
	#l2_s = [x for _, x in sorted(zip(diffs,l2), key=lambda pair: pair[0])]
	#diffs_s = [x for x, _ in sorted(zip(diffs,l2), key=lambda pair: pair[0])]
	#colors_s = [x for _, x in sorted(zip(diffs,colors), key=lambda pair: pair[0])]
	return diffs_s, l2_s, colors_s

def taught_learned(emp, diff, u):
	nz_coords = list(itertools.combinations(range(emp.shape[0]),2))
	emps = []
	diffs = []
	l2_s = []
	colors_s = []
	for (v1,v2) in nz_coords:
		diffs.append(diff[v1,v2])
		emps.append(emp[v1,v2])
	emps_s = [x for x, _ in sorted(zip(emps,diffs), key=lambda pair: pair[0])]
	diffs_s = [x for _, x in sorted(zip(emps,diffs), key=lambda pair: pair[0])]
	coords_s = [x for _, x in sorted(zip(emps,nz_coords), key=lambda pair: pair[0])]
	for (v1,v2) in coords_s:
		l2_s.append(u[v1]*u[v2])
	#l2_s = [x for _, x in sorted(zip(diffs,l2), key=lambda pair: pair[0])]
	#diffs_s = [x for x, _ in sorted(zip(diffs,l2), key=lambda pair: pair[0])]
	#colors_s = [x for _, x in sorted(zip(diffs,colors), key=lambda pair: pair[0])]
	return emps_s, diffs_s, l2_s

graphs = []

A = np.loadtxt('plots/barbell_unequal.txt')
G = nx.from_numpy_matrix(A)
#G = nx.read_gml('../data/football.gml')
_A_obs = nx.adjacency_matrix(G)
A = _A_obs.todense()

L = nx.normalized_laplacian_matrix(G).todense()
eig_vals , eig_vecs = linalg. eig ( L )
eig_list = zip( eig_vals , np . transpose ( eig_vecs ) )
eig_list . sort ( key = lambda x : x [0])
u = np.asarray([ u_i . real for u_i in eig_list [ -2][1]])[0][0]
#G = nx.grid_graph([5,5])
#mapping = dict(zip(G.nodes(),range(25)))
#G = nx.relabel_nodes(G,mapping)
#A = nx.adjacency_matrix(G).todense()
#A = A*(1/float(np.sum(A)))
#fig, ax = plt.subplots()
#norm = MidpointNormalize(midpoint=0)
#im = ax.imshow(A, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
#fig.colorbar(im)
#plt.imshow(diff, cmap='RdBu', interpolation='nearest',norm=norm)
#plt.savefig('plots/dolphins_fmm/dolphins_truth_heat.pdf')
#norm = MidpointNormalize(midpoint=0)
#walk_emp = np.loadtxt('plots/football_rw/trainingIteration_12500_expectedGraph.txt')
#fmm_emp = np.loadtxt('../dolphins_fmm_emp.txt')
#fmm_exp = genExpected_fromWalks(fmm_emp,A.sum())
#walk_exp = genExpected_fromWalks(walk_emp,A.sum())
#np.savetxt('plots/football_rw/trainingIteration_12500_expectedGraph_scaled.txt',walk_exp)





nz_1 = []
nz_2 = []
avg_1 = []
avg_2 = []
std_1 = []
std_2 = []
diffs = []
l2 = []
for i in range(1,6):
	#s='plots/dolphins_rw/trainingIteration_{}_expectedGraph.txt'.format(19*500)
	#X = (np.loadtxt(s))
	s='plots/barbell_unequal_stateless/trainingIteration_{}_expectedGraph.txt'.format(i*400)
	X_prime = (np.loadtxt(s))
	fig, ax = plt.subplots()
	norm = MidpointNormalize(midpoint=0)
	# diff_fmm = A-X_prime
	# diff_rw = A-X
	# diff_fmm_taught = fmm_exp-X_prime
	# diff_rw_taught = walk_exp-X
	im = ax.imshow(X_prime, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
	fig.colorbar(im)
	plt.savefig('plots/barbell_unequal_stateless/trainingIteration_{}_expected_heat.pdf'.format(i*400))
	plt.gcf().clear()
	# plt.gcf().clear()
	# fig, ax = plt.subplots()
	# norm = MidpointNormalize(midpoint=0)
	# im = ax.imshow(diff_rw, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
	# fig.colorbar(im)
	# plt.savefig('plots/dolphins_rw/trainingIteration_{}_expected_rw_from_truth_heat.pdf'.format(19*500))
	# plt.gcf().clear()
	# fig, ax = plt.subplots()
	# norm = MidpointNormalize(midpoint=0)
	# im = ax.imshow(diff_fmm_taught, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
	# fig.colorbar(im)
	# plt.savefig('plots/dolphins_fmm_long/trainingIteration_{}_expected_fmm_from_taught_heat.pdf'.format(i*500))
	# plt.gcf().clear()
	# plt.gcf().clear()
	# fig, ax = plt.subplots()
	# norm = MidpointNormalize(midpoint=0)
	# im = ax.imshow(diff_rw_taught, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
	# fig.colorbar(im)
	# plt.savefig('plots/dolphins_rw/trainingIteration_{}_expected_rw_from_taught_heat.pdf'.format(19*500))
	# plt.gcf().clear()

	# diff = X - X_prime
	# fig, ax = plt.subplots()
	# norm = MidpointNormalize(midpoint=0)
	# im = ax.imshow(diff, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
	# fig.colorbar(im)
	# #plt.imshow(diff, cmap='RdBu', interpolation='nearest',norm=norm)
	# plt.savefig('plots/dolphins_fmm/trainingIteration_{}_expected_diff_heat.pdf'.format(i*500))
	# plt.gcf().clear()

	# nz_coords = np.nonzero(diff)
	# nz_coords = zip(nz_coords[0],nz_coords[1])
	# X = X[X>0]
	# X_prime = X_prime[X_prime>0]
	# nz_1.append(len(X))
	# nz_2.append(len(X_prime))
	# avg_1.append(X.mean())
	# avg_2.append(X_prime.mean())
	# std_1.append(X.std())
	# std_2.append(X_prime.std())
	# colors = []
	# print('fmm')
	# diffs_s, l2_s, colors = againstFiedler(diff_fmm,u,A,1,0)
	# plt.scatter(diffs_s,l2_s,s=10, c= colors)
	# plt.savefig('plots/dolphins_fmm/trainingIteration_{}_fmm_l2_wz_plots.pdf'.format(i*500))
	# plt.gcf().clear()
	# print('rw')
	# diffs_s, l2_s, colors = againstFiedler(diff_rw,u,A,1,0)
	# plt.scatter(diffs_s,l2_s,s=10, c=colors)
	# plt.savefig('plots/dolphins_rw/trainingIteration_{}_rw_l2_wz_plots.pdf'.format(i*500))
	# plt.gcf().clear()
	# print('diff')
	# diffs_s, l2_s, colors = againstFiedler(diff,u,A,0,1)
	# plt.scatter(diffs_s,l2_s,s=10, c=colors)
	# plt.savefig('plots/dolphins_fmm/trainingIteration_{}_disc_l2_plots.pdf'.format(i*500))
	# plt.gcf().clear()
	# emp_s, diff_s, colors = taught_learned(walk_emp, diff_rw, u)
	# plt.scatter(emp_s,diff_s,s=10, c=colors,cmap='Blues')
	# plt.savefig('plots/dolphins_rw/trainingIteration_{}_rw_taught_learned_plots.pdf'.format(i*500))
	# plt.gcf().clear()
	# emp_s, diff_s, colors = taught_learned(fmm_emp, diff_fmm, u)
	# plt.scatter(emp_s,diff_s,s=10, c=colors,cmap='Blues')
	# plt.savefig('plots/dolphins_fmm/trainingIteration_{}_fmm_taught_learned_plots.pdf'.format(i*500))
	# plt.gcf().clear()
# plt.gcf().clear()
# iters = [i*500 for i in range(1,40)]
# plt.plot(iters,nz_1,'blue')
# plt.plot(iters,nz_2,'red')
# plt.savefig('plots/dolphins_fmm/nz_dist_plots.pdf')
# plt.clf()
# plt.plot(iters,avg_1,'yellow')
# plt.plot(iters,avg_2,'green')
# plt.plot(iters,std_1,'magenta')
# plt.plot(iters,std_2,'cyan')
# plt.savefig('plots/dolphins_fmm/dist_plots.pdf')
# plt.clf()

