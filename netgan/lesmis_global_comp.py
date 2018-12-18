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

def global_comp(A,X,sps_truth,spec):
    sp_emds = []
    specs = []
    ccs =[]
    for i in range(100):
        sampled_graph = gnp(X)
        sps_gen = utils.sp(sampled_graph)
        sp_emds.append(utils.emd(sps_truth,sps_gen))
        specs.append((spec-utils.specGap(sampled_graph))**2)
        ccs.append(3 * utils.statistics_triangle_count(sampled_graph) /  utils.statistics_claw_count(sampled_graph))
    stats = {}
    stats['shortest path emds']=sp_emds
    stats['spec gaps']=specs
    stats['clustering coefficients']=ccs
    print('Shortest path')
    print(np.mean(sp_emds))
    print(np.std(sp_emds))
    print('Specs')
    print(np.mean(specs))
    print(np.std(specs))
    print('Clusterig Coefficient')
    print(np.mean(ccs))
    print(np.std(ccs))
    print('True CC')
    tc = 3 * utils.statistics_triangle_count(np.array(A)) /  utils.statistics_claw_count(np.array(A))
    print(tc)
    print('ABS diff')
    print(tc-np.mean(ccs))
    return stats

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

G = nx.read_gml('../data/lesmis.gml')
_A_obs = nx.adjacency_matrix(G)
A_matrix = _A_obs.todense()
valid_edges = np.loadtxt('plots/lesmis_val_edges.txt').tolist()
valid_nonEdges = np.loadtxt('plots/lesmis_val_non_edges.txt').tolist()
valid = valid_edges+valid_nonEdges


sps_truth = utils.sp(A_matrix)
spec_truth = utils.specGap(A_matrix)


X = np.loadtxt('plots/lesmis_walk_ep/trainingIteration_2000_expectedGraph.txt')
print('Random Walks Size 16')
global_comp(A_matrix,X,sps_truth,spec_truth)
print(auc_p(X,np.asarray(A_matrix),valid))
X = np.loadtxt('plots/lemsis_edge_em_nbs/trainingIteration_500_expectedGraph.txt')
print('Random Walks Size 2 No BS adj')
global_comp(A_matrix,X,sps_truth,spec_truth)
print(auc_p(X,np.asarray(A_matrix),valid))
X = np.loadtxt('plots/lesmis_edge_wd_em_bss/trainingIteration_600_expectedGraph.txt')
print('Random Walks Size 2 BS adj')
global_comp(A_matrix,X,sps_truth,spec_truth)
print(auc_p(X,np.asarray(A_matrix),valid))







