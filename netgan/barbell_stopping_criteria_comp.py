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
import sys

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

def plot_heat(A,file_name):
    norm = MidpointNormalize(midpoint=0)
    fig, ax = plt.subplots()
    im = ax.imshow(A, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    fig.colorbar(im)
    plt.savefig(file_name)
    plt.gcf().clear()

def checkImprovement(patience, evaluation, best_so_far, max_patience, best_iter, cur_iter):
    if evaluation > best_so_far:
        patience = max_patience
        best_iter = cur_iter 
        best_so_far = evaluation 
    else:
        patience = patience-1
    return patience, best_iter, best_so_far

def evaluate_expected(X, valid, test, A_full):
    spec = utils.specGap(X)
    auc_valid,ap_valid = auc_p(X,np.asarray(A_full),valid)
    auc_test,ap_test = auc_p(X,np.asarray(A_full),test)
    return auc_test, auc_valid, ap_test, ap_valid, spec


if __name__ == "__main__":
    name = sys.argv[1]
    path = sys.argv[2]
    maxIter = int(sys.argv[3])
    specIter = int(sys.argv[4])
    apIter = int(sys.argv[5])
    aucIter = int(sys.argv[6])


    

    start = 1

    train_path = 'plots/'+name+'_training.txt'
    target_path = 'plots/'+name+'.txt'

    A_matrix = np.loadtxt(train_path)
    print('verifying correct barbell edge should be one')
    print(A_matrix[3][99])
    print(A_matrix[80][130])
    A_full = np.loadtxt(target_path)
    N = A_full.shape[0]
    
    G = nx.from_numpy_matrix(A_full)

    truth_spec = utils.specGap(A_full)

    start = 1
    
    step = 400
    k=maxIter/step

    #initialize all params

    

    barbell_edge_density_1 = []
    barbell_edge_density_2 = []



    num_edges_crossing_1 = []
    num_edges_crossing_2 = []
    num_edges_crossing_3 = []



    for i in range(start,k+1):
        iterNum = i*step
        print(iterNum)


        X_1 = np.loadtxt(path+'/samples_{}.txt'.format(iterNum))
        X_1 = genExpected_fromWalks(X_1,A_full.sum())


        barbell_edge_density_1.append(X_1[3][99])
        barbell_edge_density_2.append(X_1[80][130])


        num_edges_crossing_1.append(X_1[np.ix_(range(50),range(50,100))].sum())
        num_edges_crossing_2.append(X_1[np.ix_(range(50,100),range(100,150))].sum())
        num_edges_crossing_3.append(X_1[np.ix_(range(50),range(100,150))].sum())

        print(X_1[np.ix_(range(50),range(50,100))].sum())
        print(X_1[np.ix_(range(50,100),range(100,150))].sum())
        print(X_1[np.ix_(range(50),range(100,150))].sum())


        

    index = [i*step for i in range(start,k+1)]

    #Barbell Edge Density
    plt.axvline(x=specIter,color='r', linestyle='dashed',label='Spectrum Stop')
    plt.axvline(x=apIter, color='b', linestyle='dashed',label='AP Stop')
    plt.axvline(x=aucIter, color='g',linestyle='dashed',label='AUC Stop')
    plt.plot(index,barbell_edge_density_1)
    plt.plot(index,barbell_edge_density_2)
    plt.xlabel('Num Training Iterations')
    plt.ylabel('Edge Density Expectation Matrix')
    plt.legend(loc='best')
    plt.savefig(path+'/edgeDensity_expected_comp.pdf')
    plt.gcf().clear()


    plt.axvline(x=specIter,color='r', linestyle='dashed',label='Spectrum Stop')
    plt.axvline(x=apIter, color='b', linestyle='dashed',label='AP Stop')
    plt.axvline(x=aucIter, color='g',linestyle='dashed',label='AUC Stop')
    plt.plot(index,num_edges_crossing_1, label='Cross 1-2')
    plt.plot(index,num_edges_crossing_2, label='Cross 2-3')
    plt.plot(index,num_edges_crossing_3, label='Cross 1-3')
    plt.legend(loc='best')
    plt.xlabel('Num Training Iterations')
    plt.ylabel('Num Edges Crossing')
    plt.savefig(path+'/num_edges_crossing_expected_comp.pdf')
    plt.gcf().clear()

    
    plt.plot(index[-6:],num_edges_crossing_1[-6:], label='Cross 1-2')
    plt.plot(index[-6:],num_edges_crossing_2[-6:], label='Cross 2-3')
    plt.plot(index[-6:],num_edges_crossing_3[-6:], label='Cross 1-3')
    plt.legend(loc='best')
    plt.xlabel('Num Training Iterations')
    plt.ylabel('Num Edges Crossing')
    plt.savefig(path+'/num_edges_crossing_expected_comp_last.pdf')
    plt.gcf().clear()

    



