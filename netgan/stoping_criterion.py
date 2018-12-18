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






if __name__ == "__main__":
    name = sys.argv[1]
    path = sys.argv[2]
    maxIter = sys.argv[3]
    print(name)
    print(path)
    print(maxIter)
    train_path = 'plots/'+name+'_training.txt'
    target_path = 'plots/'+name+'.txt'

    A_matrix = np.loadtxt(train_path)
    A_full = np.loadtxt(target_path)
    N = A_full.shape[0]

    valid_edges = np.loadtxt('plots/'+name+'_val_edges.txt').tolist()
    valid_nonEdges = np.loadtxt('plots/'+name+'_val_non_edges.txt').tolist()
    valid = valid_edges+valid_nonEdges
    test_edges = np.loadtxt('plots/'+name+'_test_edges.txt').tolist()
    test_nonEdges = np.loadtxt('plots/'+name+'_test_non_edges.txt').tolist()
    test = test_edges+test_nonEdges

    G = nx.from_numpy_matrix(A_full)

    truth_spec = utils.specGap(A_full)


    start = 1
    
    step = 400
    k=int(maxIter)/step
    max_patience = 5
    patience_spec = max_patience
    patience_auc = max_patience
    patience_ep = max_patience
    patience_vc = max_patience
    best_spec = -10
    best_auc = 0
    best_ep = 0
    best_vc = 0
    best_spec_iter = step
    best_auc_iter = step
    best_ep_iter = step
    best_vc_iter = step
    for i in range(start,k+1):
        iterNum = i*step
        print('new step')
        print(i)
        print(i*step)

        X_w = np.loadtxt(path+'/samples_{}.txt'.format(iterNum))
        X_w = genExpected_fromWalks(X_w,A_full.sum())

        #lstm_grads = np.loadtxt(path+'/lstm_weights_grads_{}.txt'.format(iterNum))
        w_up_grads = np.loadtxt(path+'/w_up_grads_{}.txt'.format(iterNum))
        w_down_grads = np.loadtxt(path+'/w_down_grads_{}.txt'.format(iterNum))

        plot_heat(X_w,path+'/expected_heat_{}.jpg'.format(iterNum))
        #plot_heat(lstm_grads,path+'/lstm_grad_heat_{}.jpg'.format(iterNum))
        plot_heat(w_up_grads,path+'/w_up_grad_heat_{}.jpg'.format(iterNum))
        plot_heat(w_down_grads,path+'/w_down_grad_heat_{}.jpg'.format(iterNum))

        cur_spec = utils.specGap(X_w)
        print(cur_spec)

        
        auc_w_valid,ep_w_valid = auc_p(X_w,np.asarray(A_full),valid)
        print(auc_w_valid)
        if patience_spec > 0:
            patience_spec, best_spec_iter, best_spec = checkImprovement(patience_spec, -1*(abs(cur_spec-truth_spec)), best_spec, max_patience, best_spec_iter, iterNum)
        if patience_auc > 0:
            patience_auc, best_auc_iter, best_auc = checkImprovement(patience_auc, auc_w_valid, best_auc, max_patience, best_auc_iter, iterNum)
        if patience_ep > 0:
            patience_ep, best_ep_iter, best_ep = checkImprovement(patience_ep, ep_w_valid, best_ep, max_patience, best_ep_iter, iterNum)
        if patience_vc > 0:
            patience_vc, best_vc_iter, best_vc = checkImprovement(patience_vc, auc_w_valid+ep_w_valid, best_vc, max_patience, best_vc_iter, iterNum)
        print(patience_auc)
        print(patience_ep)
        print(patience_spec)


    print('true spec')
    print(truth_spec)
    X_w = np.loadtxt(path+'/samples_{}.txt'.format(best_spec_iter))
    X_w = genExpected_fromWalks(X_w,A_full.sum())
    np.savetxt(path+'/spec_stop_model.txt',X_w)

    X_w = np.loadtxt(path+'/samples_{}.txt'.format(best_auc_iter))
    X_w = genExpected_fromWalks(X_w,A_full.sum())
    np.savetxt(path+'/auc_stop_model.txt',X_w)

    X_w = np.loadtxt(path+'/samples_{}.txt'.format(best_ep_iter))
    X_w = genExpected_fromWalks(X_w,A_full.sum())
    np.savetxt(path+'/ep_stop_model.txt',X_w)

    X_w = np.loadtxt(path+'/samples_{}.txt'.format(best_vc_iter))
    X_w = genExpected_fromWalks(X_w,A_full.sum())
    np.savetxt(path+'/vc_stop_model.txt',X_w)

    best_iters = {}
    best_iters['best_spec_iter']=best_spec_iter
    best_iters['best_auc_iter']=best_auc_iter
    best_iters['best_ep_iter']=best_ep_iter
    best_iters['best_vc_iter']=best_vc_iter
    best_iters['best_spec']=best_spec
    best_iters['best_auc']=best_auc
    best_iters['best_ep']=best_ep
    best_iters['best_vc']=best_vc

    f = open(path+'/stopping_criterion_numIters.txt',"w")
    f.write( str(best_iters) )
    f.close()






        