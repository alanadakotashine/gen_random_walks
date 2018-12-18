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

def l2_exp_weight(x,y):
    n = len(x)
    distance = 0
    for i in range(n):
        weight = np.exp(-1*i)
        distance = distance+(((x[i]-y[i])**2)*weight)
    return distance

def l2_lin_weight(x,y):
    n = len(x)
    distance = 0
    for i in range(n):
        weight = 1.0/float(i+1)
        distance = distance+(((x[i]-y[i])**2)*weight)
    return distance

def graphEval(X,truth_spec,true_sp,true_cc,true_dd,true_bc):
    sp_emd_cur = []
    cc_emd_cur = []
    dd_emd_cur = []
    assorts_cur = []
    spec_l2_cur = []
    spec_l2_lin_cur = []
    bc_emd_cur = []

    for j in range(20):
        A = gnp(X)
        G = nx.from_numpy_matrix(A)
        print(nx.is_connected(G))
        if not nx.is_connected(G):
            Gc = max(nx.connected_component_subgraphs(G), key=len)
            print(len(Gc.nodes()))
        sp = utils.sp(A)
        cc = utils.cc(A)
        dd = utils.degree_sequence(A)
        spec_weight_l2 = l2_exp_weight(truth_spec,utils.spectrum(A))
        spec_weight_l2_lin = l2_lin_weight(truth_spec,utils.spectrum(A))
        bc = sorted(nx.betweenness_centrality(G).values())



        sp_emd_cur.append(utils.emd(sp,true_sp))
        cc_emd_cur.append(utils.emd(cc,true_cc))
        dd_emd_cur.append(utils.emd(dd,true_dd))
        assorts_cur.append(nx.degree_assortativity_coefficient(G))
        spec_l2_cur.append(spec_weight_l2)
        spec_l2_lin_cur.append(spec_weight_l2_lin)
        bc_emd_cur.append(utils.emd(bc,true_bc))

    return np.mean(sp_emd_cur), np.mean(cc_emd_cur), np.mean(dd_emd_cur), np.mean(assorts_cur), np.mean(spec_l2_cur), np.mean(bc_emd_cur), np.mean(spec_l2_lin_cur)


if __name__ == "__main__":
    name = sys.argv[1]
    path_1 = sys.argv[2]
    path_2 = sys.argv[3]
    lastIter_1 = int(sys.argv[4])
    lastIter_2 = int(sys.argv[5])
    label_1 = sys.argv[6]
    label_2 = sys.argv[7]

    maxIter = max(lastIter_1,lastIter_2)

    start = 1

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

    G_true = nx.from_numpy_matrix(A_full)

    truth_spec = utils.spectrum(A_full)
    truth_spec_gap = utils.specGap(A_full)

    start = 5
    
    step = 400
    k=maxIter/step

    #initialize all params

    aucs_test_1 = []
    aucs_valid_1 = []
    aps_test_1 = []
    aps_valid_1 = []
    specs_1 = []

    aucs_test_2 = []
    aucs_valid_2 = []
    aps_test_2 = []
    aps_valid_2 = []
    specs_2 = []

    true_spec_gaps = []

    sp_emds_1 = []
    cc_emds_1 = []
    dd_emds_1 = []
    assorts_1 = []
    spectrum_weighted_distances_1 = []
    bc_emds_1 = []
    sp_emds_2 = []
    cc_emds_2 = []
    dd_emds_2 = []
    assorts_2 = []
    spectrum_weighted_distances_2 = []
    bc_emds_2 = []
    spec_dist_lin_1 = []
    spec_dist_lin_2=[]



    true_sp = utils.sp(A_full)
    true_cc = utils.cc(A_full)
    true_dd = utils.degree_sequence(A_full)

    true_assort = nx.degree_assortativity_coefficient(G_true)

    true_bc = sorted(nx.betweenness_centrality(G_true))

    true_assorts = []



    for i in range(start,k+1):
        iterNum_cur = i*step
        print('iternum')
        print(iterNum_cur)
        if iterNum_cur > lastIter_1:
            iterNum = lastIter_1
        else:
            iterNum = iterNum_cur

        X_1 = np.loadtxt(path_1+'/samples_{}.txt'.format(iterNum))
        X_1 = genExpected_fromWalks(X_1,A_full.sum())

        if iterNum_cur > lastIter_2:
            iterNum = lastIter_2
        else:
            iterNum = iterNum_cur

        X_2 = np.loadtxt(path_2+'/samples_{}.txt'.format(iterNum))
        X_2 = genExpected_fromWalks(X_2,A_full.sum())

        auc_test_1, auc_valid_1, ap_test_1, ap_valid_1, spec_1 = evaluate_expected(X_1, valid, test, A_full)
        auc_test_2, auc_valid_2, ap_test_2, ap_valid_2, spec_2 = evaluate_expected(X_2, valid, test, A_full)

        aucs_test_1.append(auc_test_1)
        aucs_valid_1.append(auc_valid_1)
        aps_test_1.append(ap_test_1)
        aps_valid_1.append(ap_valid_1)
        specs_1.append(spec_1)

        aucs_test_2.append(auc_test_2)
        aucs_valid_2.append(auc_valid_2)
        aps_test_2.append(ap_test_2)
        aps_valid_2.append(ap_valid_2)
        specs_2.append(spec_2)

        true_spec_gaps.append(truth_spec_gap)
        true_assorts.append(true_assort)


        print(label_1)
        mean_sp_1, mean_cc_1, mean_dd_1, mean_assort_1, mean_spec_l2_1, mean_bc_1, mean_spec_l2_lin_1 = graphEval(X_1,truth_spec,true_sp,true_cc,true_dd,true_bc)
        print(label_2)
        mean_sp_2, mean_cc_2, mean_dd_2, mean_assort_2, mean_spec_l2_2, mean_bc_2,  mean_spec_l2_lin_2 = graphEval(X_1,truth_spec,true_sp,true_cc,true_dd,true_bc)


        sp_emds_1.append(mean_sp_1)
        cc_emds_1.append(mean_cc_1)
        dd_emds_1.append(mean_dd_1)
        assorts_1.append(mean_assort_1)
        spectrum_weighted_distances_1.append(mean_spec_l2_1)
        bc_emds_1.append(mean_bc_1)
        spec_dist_lin_1.append(mean_spec_l2_lin_1)

        sp_emds_2.append(mean_sp_2)
        cc_emds_2.append(mean_cc_2)
        dd_emds_2.append(mean_dd_2)
        assorts_2.append(mean_assort_2)
        spectrum_weighted_distances_2.append(mean_spec_l2_2)
        bc_emds_2.append(mean_bc_2)
        spec_dist_lin_2.append(mean_spec_l2_lin_2)

    index = [i*step for i in range(start,k+1)]

    #Spectrums
    plt.plot(index,specs_1,label=label_1)
    plt.plot(index,specs_2,label=label_2)
    plt.plot(index,true_spec_gaps,label='Spec Gap Truth')
    plt.xlabel('Num Training Iterations')
    plt.ylabel('Spectral Gap of Generated Expectation Matrix')
    plt.legend(loc='best')
    plt.savefig('plots/'+name+'specGaps_expected_comp.pdf')
    plt.gcf().clear()

    plt.plot(index,aucs_valid_1,label=label_1)
    plt.plot(index,aucs_valid_2,label=label_2)
    plt.legend(loc='best')
    plt.xlabel('Num Training Iterations')
    plt.ylabel('ROC AUC Score')
    plt.savefig('plots/'+name+'auc_valid_expected_comp.pdf')
    plt.gcf().clear()

    plt.plot(index,aucs_test_1,label=label_1)
    plt.plot(index,aucs_test_2,label=label_2)
    plt.legend(loc='best')
    plt.xlabel('Num Training Iterations')
    plt.ylabel('ROC AUC Score')
    plt.savefig('plots/'+name+'auc_test_expected_comp.pdf')
    plt.gcf().clear()

    plt.plot(index,aps_valid_1,label=label_1)
    plt.plot(index,aps_valid_2,label=label_2)
    plt.legend(loc='best')
    plt.xlabel('Num Training Iterations')
    plt.ylabel('Average Precision Score')
    plt.savefig('plots/'+name+'ap_valid_expected_comp.pdf')
    plt.gcf().clear()

    plt.plot(index,aps_test_1,label=label_1)
    plt.plot(index,aps_test_2,label=label_2)
    plt.legend(loc='best')
    plt.xlabel('Num Training Iterations')
    plt.ylabel('Average Precision Score')
    plt.savefig('plots/'+name+'ap_test_expected_comp.pdf')
    plt.gcf().clear()

    #Shortest Path


    plt.plot(index,sp_emds_1,label=label_1)
    plt.plot(index,sp_emds_2,label=label_2)
    plt.xlabel('Num Training Iterations')
    plt.ylabel('EMD Shortest Path')
    plt.legend(loc='best')
    plt.savefig('plots/'+name+'_shortestPath_emd_comp.pdf')
    plt.gcf().clear()

    #Clustering Coefficient


    plt.plot(index,cc_emds_1,label=label_1)
    plt.plot(index,cc_emds_2,label=label_2)
    plt.xlabel('Num Training Iterations')
    plt.ylabel('EMD Clustering Coefficient')
    plt.legend(loc='best')
    plt.savefig('plots/'+name+'_clusteringCoefficient_emd_comp.pdf')
    plt.gcf().clear()

    #Degree Distribution


    plt.plot(index,dd_emds_1,label=label_1)
    plt.plot(index,dd_emds_2,label=label_2)
    plt.xlabel('Num Training Iterations')
    plt.ylabel('EMD Degree Sequence')
    plt.legend(loc='best')
    plt.savefig('plots/'+name+'_degreeSequence_emd_comp.pdf')
    plt.gcf().clear()

    #Assortativity


    plt.plot(index,assorts_1,label=label_1)
    plt.plot(index,assorts_2,label=label_2)
    plt.plot(index,true_assorts,label='True Assortativity')
    plt.xlabel('Num Training Iterations')
    plt.ylabel('Degree Assortativity')
    plt.legend(loc='best')
    plt.savefig('plots/'+name+'_assorativity_comp.pdf')
    plt.gcf().clear()

    #Spectrum L2 Distances


    plt.plot(index,spectrum_weighted_distances_1,label=label_1)
    plt.plot(index,spectrum_weighted_distances_2,label=label_2)
    plt.xlabel('Num Training Iterations')
    plt.ylabel('Exponential Weighted L2 Norm Spectrum')
    plt.legend(loc='best')
    plt.savefig('plots/'+name+'_spectrum_weighted_l2_comp.pdf')
    plt.gcf().clear()

    plt.plot(index,spec_dist_lin_1,label=label_1)
    plt.plot(index,spec_dist_lin_2,label=label_2)
    plt.xlabel('Num Training Iterations')
    plt.ylabel('Linear Weighted L2 Norm Spectrum')
    plt.legend(loc='best')
    plt.savefig('plots/'+name+'_spectrum_weighted_l2_linear_comp.pdf')
    plt.gcf().clear()

    #Spectrum L2 Distances
    

    plt.plot(index,bc_emds_1,label=label_1)
    plt.plot(index,bc_emds_2,label=label_2)
    plt.xlabel('Num Training Iterations')
    plt.ylabel('EMD Betweeness Centrality')
    plt.legend(loc='best')
    plt.savefig('plots/'+name+'_btwn_centrality_comp.pdf')
    plt.gcf().clear()



