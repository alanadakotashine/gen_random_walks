import tensorflow as tf
import utils
import scipy.sparse as sp
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
import time
from matplotlib.colors import Normalize
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

def normDensity(edge_probs,targetSum):
        #edge_probs = gr_edgeProb.data
        #assuming that all edge probs were included 
        #n = gr_edgeProb.shape[0]
        #edge_probs = gr_edgeProb.todense()
        np.fill_diagonal(edge_probs, 0)
        edge_probs = (edge_probs / np.sum(edge_probs))*targetSum
        edge_probs[edge_probs > 1]=1
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
        if z[v1,v2] > .8:
            print('large diff')
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
    #       if colors_s[-1]=='r':
    #           print('error')
        k=k+1
    #l2_s = [x for _, x in sorted(zip(diffs,l2), key=lambda pair: pair[0])]
    #diffs_s = [x for x, _ in sorted(zip(diffs,l2), key=lambda pair: pair[0])]
    #colors_s = [x for _, x in sorted(zip(diffs,colors), key=lambda pair: pair[0])]
    return diffs_s, l2_s, colors_s

def gnp(X):
    X = np.triu(X)
    G = np.random.binomial(1,p=X)
    G = G+G.T
    return G

def spec_gap_largest_component(A):
    G = nx.from_numpy_matrix(A)
    Gc = max(nx.connected_component_subgraphs(G), key=len)
    A = np.asarray(nx.adjacency_matrix(Gc).todense())
    return utils.specGap(A)

def edge_discs(X,A):
    eos = []
    ems = []
    numEdge = float(A[A==1].shape[0])
    numNonEdge = float(A[A==0].shape[0])
    for i in range(20):
        graph = gnp(X)
        eo = ((A == graph) & (A == 1)).sum()
        em = ((A == graph) & (A<1)).sum()
        eos.append(float(eo)/numEdge)
        ems.append(float(em)/numNonEdge)
    return np.mean(eos), np.std(eos), np.mean(ems), np.std(ems)

def edge_nonedge_scores(X,A,valid=None):
    N = np.shape(A)[0]
    edgeScores= []
    nonedgeScores = []
    if valid==None:
        us = range(N)
        vs=range(N)
    else:
        us = [int(x) for [x,y] in valid]
        vs = [int(y) for [x,y] in valid]
    for i in us:
        for j in vs:
            if A[i][j]==1:
                edgeScores.append(X[i][j])
            else:
                nonedgeScores.append(X[i][j])
    return np.mean(edgeScores), np.std(edgeScores), np.mean(nonedgeScores), np.std(nonedgeScores)



def auc_p(X,A,valid):
    eos = []
    ems = []
    numEdge = float(A[A==1].shape[0])
    numNonEdge = float(A[A==0].shape[0])
    trueScores = []
    predScores = []
    for [u,v] in valid:
        trueScores.append(A[int(u)][int(v)])
        predScores.append(X[int(u)][int(v)])
    auc = roc_auc_score(trueScores,predScores)
    ap = average_precision_score(trueScores,predScores)
    return auc,ap






#G = nx.gnp_random_graph(50,.95)
#H = nx.gnp_random_graph(50,.5)
#H = nx.relabel_nodes(H,dict(zip(range(50),range(50,100))))
#G = nx.union(H,G)
#G.add_edge(3,99)
#G = nx.read_gml('../data/barbell_unequal.gml')
#_A_obs = nx.adjacency_matrix(G)
#A_full = _A_obs.todense()
A_matrix = np.loadtxt('plots/barbell_unequal_training.txt')
A_full = np.loadtxt('plots/barbell_unequal.txt')
N = A_matrix.shape[0]

valid_edges = np.loadtxt('plots/barbell_unequal_val_edges.txt').tolist()
valid_nonEdges = np.loadtxt('plots/barbell_unequal_val_non_edges.txt').tolist()
valid = valid_edges+valid_nonEdges
G = nx.from_numpy_matrix(A_matrix)

L = nx.normalized_laplacian_matrix(G).todense()
eig_vals , eig_vecs = linalg. eig ( L )
eig_list = zip( eig_vals , np . transpose ( eig_vecs ) )
eig_list . sort ( key = lambda x : x [0])


u = np.asarray([ u_i . real for u_i in eig_list [ -2][1]])[0][0]


truth = utils.compute_graph_statistics(np.asarray(A_matrix))
f = open('plots/truth.txt',"w")
f.write( str(truth) )
f.close()

truth_spec = utils.specGap(A_full)
train_spec = utils.specGap(A_matrix)

truth_cc = utils.cc(A_full)



cc_emd_combo = []
cc_emd_reg = []
cc_emd_fmm = []
cc_emd_combo_std = []
cc_emd_reg_std = []
cc_emd_fmm_std = []
k=11
for i in range(1,k):
    print(i)
    X_c = np.loadtxt('plots/barbell_sameDensity/barbell_combo_mixed/trainingIteration_{}_expectedGraph.txt'.format(i*100))
    X_f = np.loadtxt('plots/barbell_sameDensity/barbell_fmm/trainingIteration_{}_expectedGraph.txt'.format(i*100))
    X_r = np.loadtxt('plots/barbell_sameDensity/barbell_walk_mixed/trainingIteration_{}_expectedGraph.txt'.format(i*100))
    
    cc_emd_combo_iter = []
    cc_emd_fmm_iter = []
    cc_emd_reg_iter = []
    for j in range(100):
        sampled_graph_c = gnp(X_c)
        sampled_graph_f = gnp(X_f)
        sampled_graph_r = gnp(X_r)
        cc_c = utils.cc(sampled_graph_c)
        cc_f = utils.cc(sampled_graph_f)
        cc_r = utils.cc(sampled_graph_r)
        cc_emd_combo_iter.append(utils.emd(truth_cc,cc_c))
        cc_emd_fmm_iter.append(utils.emd(truth_cc,cc_f))
        cc_emd_reg_iter.append(utils.emd(truth_cc,cc_r))
    cc_emd_combo.append(np.mean(cc_emd_combo_iter))
    cc_emd_fmm.append(np.mean(cc_emd_fmm_iter))
    cc_emd_reg.append(np.mean(cc_emd_reg_iter))
    cc_emd_combo_std.append(np.std(cc_emd_combo_iter))
    cc_emd_fmm_std.append(np.std(cc_emd_fmm_iter))
    cc_emd_reg_std.append(np.std(cc_emd_reg_iter))



plt.errorbar(range(1,k),cc_emd_combo,yerr=cc_emd_combo_std,label='Combo')
plt.errorbar(range(1,k),cc_emd_fmm,yerr=cc_emd_fmm_std,label='FMM')
plt.errorbar(range(1,k),cc_emd_reg,yerr=cc_emd_reg_std,label='Reg Walk')
plt.legend(loc='best')
plt.savefig('plots/barbell_sameDensity/barbell_cc_comp.pdf')
plt.gcf().clear()
