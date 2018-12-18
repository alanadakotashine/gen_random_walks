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





#G = nx.gnp_random_graph(50,.95)
#H = nx.gnp_random_graph(50,.5)
#H = nx.relabel_nodes(H,dict(zip(range(50),range(50,100))))
#G = nx.union(H,G)
#G.add_edge(3,99)
#G = nx.read_gml('../data/three_cluster_line.gml')
#G = nx.three_cluster_line_club_graph()
#_A_obs = nx.adjacency_matrix(G)
#A_full = _A_obs.todense()
A_matrix = np.loadtxt('plots/three_cluster_line_training.txt')
A_full = np.loadtxt('plots/three_cluster_line.txt')
N = A_full.shape[0]

valid_edges = np.loadtxt('plots/three_cluster_line_val_edges.txt').tolist()
valid_nonEdges = np.loadtxt('plots/three_cluster_line_val_non_edges.txt').tolist()
valid = valid_edges+valid_nonEdges
test_edges = np.loadtxt('plots/three_cluster_line_test_edges.txt').tolist()
test_nonEdges = np.loadtxt('plots/three_cluster_line_test_non_edges.txt').tolist()
test = test_edges+test_nonEdges
G = nx.from_numpy_matrix(A_full)

L = nx.normalized_laplacian_matrix(G).todense()
eig_vals , eig_vecs = linalg. eig ( L )
eig_list = zip( eig_vals , np . transpose ( eig_vecs ) )
eig_list . sort ( key = lambda x : x [0])


u = np.asarray([ u_i . real for u_i in eig_list [ -2][1]])[0][0]


truth = utils.compute_graph_statistics(np.asarray(A_full))
f = open('plots/truth.txt',"w")
f.write( str(truth) )
f.close()

truth_spec = utils.specGap(A_full)

#train_spec = utils.specGap(A_matrix)



spec_gaps_w = []
spec_gaps_e = []
spec_gaps_w_raw = []
spec_gaps_e_raw = []
spec_gaps_a_raw = []
spec_gaps_a = []
spec_gaps_w_std = []
spec_gaps_e_std = []
spec_gaps_a_std=[]
spec_gaps_truth = []
spec_gaps_train = []
spec_gaps_w_emp = []
spec_gaps_e_emp = []
spec_gaps_a_emp = []
cross_w_emp = []
cross_e_emp = []
cross_a_emp = []
cross_w_emp_std = []
cross_e_emp_std = []
cross_a_emp_std = []
points = [500,1000,1500,1900]
eo_means = []
eo_stds = []
em_means = []
em_stds = []
eo_means_f = []
eo_stds_f = []
em_means_f = []
em_stds_f = []
es_means = []
es_stds = []
nes_means = []
nes_stds = []
es_v_means = []
es_v_stds = []
nes_v_means = []
nes_v_stds = []
aucs_w_valid = []
aps_w_valid = []
aucs_e_valid = []
aps_e_valid = []
aucs_w_test = []
aps_w_test = []
aucs_e_test = []
aps_e_test = []
num_edges_crossing = []
num_edges_crossing_e = []
num_edges_crossing_2 = []
num_edges_crossing_e_2 = []
num_edges_crossing_3= []
num_edges_crossing_e_3 = []
three_cluster_line_edge_density_w = []
three_cluster_line_edge_density_e = []
three_cluster_line_edge_density_w_2 = []
three_cluster_line_edge_density_e_2 = []
three_cluster_line_edge_density_a = []
start = 1
k=11
step = 400
max_patience_spec = 5
max_patience_auc = 5
max_partience_ep = 5
best_spec = -10
best_auc = 0
best_ep = 0
best_spec_iter = step
best_auc_iter = step
best_ep_iter = step
for i in range(start,k):
    print(i)
    print(i*step)
    X_e = np.loadtxt('plots/three_cluster_line/walk_stateless_40layers_disciter_3/samples_{}.txt'.format(i*step))
    X_e = genExpected_fromWalks(X_e,A_full.sum())
    #X_a = np.loadtxt('plots/fmm_three_cluster_line_stateless_disc_iter_1/samples_{}.txt'.format(i*step))
    #R_a = normDensity(np.loadtxt('plots/three_cluster_line_stateless_keephidden_mixed/trainingIteration_{}_sampledEdgeDensity.txt'.format(i*100)), A_full.sum())
    #if i < 11:
    X_w = np.loadtxt('plots/three_cluster_line/three_cluster_wstate_walk_disc_iter_1/samples_{}.txt'.format(i*step))
    X_w = genExpected_fromWalks(X_w,A_full.sum())

    #plot_heat(X_e,'plots/three_cluster_line_stateless_disc_iter_1_/expected_heat_{}.jpg'.format(i*step))
    #plot_heat(X_w,'plots/expected_heat_{}.jpg'.format(i*step))

    # print('num edges')
    # print(A_full.sum())
    # print('density')
    # p=float(A_full.sum())/float(N*N)
    # print(p)
    #print('uniform crossing')
    # print(p*900)
    #X_a = genExpected_fromWalks(X_a,A_full.sum())
    #R_e = normDensity(np.loadtxt('plots/three_cluster_line_stateless_keephidden/trainingIteration_{}_sampledEdgeDensity.txt'.format(i*100)), A_full.sum())
    #R_w = normDensity(np.loadtxt('plots/three_cluster_line_walk_mixed/trainingIteration_{}_sampledEdgeDensity.txt'.format(i*100)), A_full.sum())
    spec_gaps_w.append(utils.specGap(X_w))
    spec_gaps_e.append(utils.specGap(X_e))
    #spec_gaps_w.append(utils.specGap(X_w))
    #print('disc iter 1 spec gap')
    #temp = utils.specGap(X_e)
    #print(temp)
    print('disc iter 1')
    temp = utils.specGap(X_w)
    print(temp)
    print('spec truth, spec now')
    print(truth_spec)
    print(temp)
    # spec_gaps_a.append(temp)
    # # spec_gaps_a_raw.append(utils.specGap(R_a))
    # # spec_gaps_e_raw.append(utils.specGap(R_e))
    auc_e_valid,ap_e_valid = auc_p(X_e,np.asarray(A_full),valid)
    auc_w_valid,ap_w_valid = auc_p(X_w,np.asarray(A_full),valid)
    auc_e_test,ap_e_test = auc_p(X_e,np.asarray(A_full),test)
    auc_w_test,ap_w_test = auc_p(X_w,np.asarray(A_full),test)
    # print('auc')
    # print(auc_w_test)
    # print('ap')
    # print(ap_w_test)
    # # auc_a,ap_a = auc_p(X_a,np.asarray(A_full),valid)
    three_cluster_line_edge_density_e.append(X_e[3][99])
    three_cluster_line_edge_density_w.append(X_w[3][99])
    three_cluster_line_edge_density_e_2.append(X_e[80][130])
    three_cluster_line_edge_density_w_2.append(X_w[80][130])
    num_edges_crossing.append(X_w[np.ix_(range(50),range(50,100))].sum())
    num_edges_crossing_e.append(X_e[np.ix_(range(50),range(50,100))].sum())
    num_edges_crossing_2.append(X_w[np.ix_(range(50,100),range(100,150))].sum())
    num_edges_crossing_e_2.append(X_e[np.ix_(range(50,100),range(100,150))].sum())
    num_edges_crossing_3.append(X_w[np.ix_(range(50),range(100,150))].sum())
    num_edges_crossing_e_3.append(X_e[np.ix_(range(50),range(100,150))].sum())
    # # # # num_edges_crossing_raw.append(R_w[np.ix_(range(50),range(50,100))].sum())
    # # # # num_edges_crossing_e_raw.append(R_e[np.ix_(range(50),range(50,100))].sum())
    # # # num_edges_crossing_a.append(X_a[np.ix_(range(50),range(50,100))].sum())
    # print('should be one')
    # temp = X_w[np.ix_(range(50),range(50,100))].sum()
    # print(temp)
    temp = X_w[np.ix_(range(50,100),range(100,150))].sum()
    print(temp)
    temp = X_w[np.ix_(range(50),range(50,100))].sum()
    print(temp)
    temp = X_w[np.ix_(range(50),range(100,150))].sum()
    print(temp)
    print('clusters')
    temp = X_w[np.ix_(range(50),range(50))].sum()
    print(temp)
    temp = X_w[np.ix_(range(50,100),range(50,100))].sum()
    print(temp)
    temp = X_w[np.ix_(range(100,150),range(100,150))].sum()
    print(temp)
    # # # print(temp)
    # # print('should be zero')
    # # temp = X_w[np.ix_(range(50,100),range(100,150))].sum()
    # # print(temp)
    # # print('should be one')
    # # print(X_w[3][99])
    # # print(X_w[80][130])
    # # temp = X_w[np.ix_(range(30),range(90,120))].sum()
    # # print(temp)
    # # temp = X_w[np.ix_(range(30,60),range(90,120))].sum()
    # #print(temp)
    # #num_edges_crossing_a.append(X_a[np.ix_(range(50),range(50,100))].sum())
    # # # eo_mean, eo_std, em_mean, em_std = edge_discs(X_w,A_matrix)
    # # # es_mean, es_std, nes_mean, nes_std = edge_nonedge_scores(X_w,np.asarray(A_full))
    # # # es_v_mean, es_v_std, nes_v_mean, nes_v_std = edge_nonedge_scores(X_w,np.asarray(A_full),valid)
    # # # eo_mean_f, eo_std_f, em_mean_f, em_std_f = edge_discs(X_w,np.asarray(A_full))
    # # # eo_means.append(eo_mean)
    # # # eo_stds.append(eo_std)
    # # # em_means.append(em_mean)
    # # # em_stds.append(em_std)
    # # # es_means.append(es_mean)
    # # # es_stds.append(es_std)
    # # # nes_means.append(nes_mean)
    # # # nes_stds.append(nes_std)
    # # # es_v_means.append(es_v_mean)
    # # # es_v_stds.append(es_v_std)
    # # # nes_v_means.append(nes_v_mean)
    # # # nes_v_stds.append(nes_v_std)
    # # # eo_means_f.append(eo_mean_f)
    # # # eo_stds_f.append(eo_std_f)
    # # # em_means_f.append(em_mean_f)
    # # # em_stds_f.append(em_std_f)
    aucs_w_valid.append(auc_w_valid)
    aps_w_valid.append(ap_w_valid)
    aucs_e_valid.append(auc_e_valid)
    aps_e_valid.append(ap_e_valid)

    aucs_w_test.append(auc_w_test)
    aps_w_test.append(ap_w_test)
    aucs_e_test.append(auc_e_test)
    aps_e_test.append(ap_e_test)

    # # # #X_e = np.loadtxt('plots/three_cluster_line_edge_wd_em/trainingIteration_{}_expectedGraph.txt'.format(i*500))
    # # # #X_a = np.loadtxt('plots/trainingIteration_{}_expectedGraph.txt'.format(i*400))
    # # # spec_gaps_w_iter = []
    # # # spec_gaps_e_iter = []
    # # # spec_gaps_a_iter = []
    # # # cross_w_iter = []
    # # # cross_e_iter = []
    # # # cross_a_iter = []
    # # # for j in range(10):
    # # #     sampled_graph_w = gnp(X_w)
    # # #     sampled_graph_e = gnp(X_e)
    # # #     #sampled_graph_a = gnp(X_a)
    # # #     spec_gaps_w_iter.append(spec_gap_largest_component(sampled_graph_w))
    # # #     spec_gaps_e_iter.append(spec_gap_largest_component(sampled_graph_e))
    # # #     #spec_gaps_a_iter.append(spec_gap_largest_component(sampled_graph_a))
    # # #     cross_w_iter.append(sampled_graph_w[np.ix_(range(50),range(50,100))].sum())
    # # #     cross_e_iter.append(sampled_graph_e[np.ix_(range(50),range(50,100))].sum())
    # # #     #cross_a_iter.append(sampled_graph_a[np.ix_(range(50),range(50,100))].sum())
    # # # spec_gaps_w_emp.append(np.mean(spec_gaps_w_iter))
    # # # spec_gaps_e_emp.append(np.mean(spec_gaps_e_iter))
    # # # #spec_gaps_a_emp.append(np.mean(spec_gaps_a_iter))
    # # # cross_w_emp.append(np.mean(cross_w_iter))
    # # # cross_e_emp.append(np.mean(cross_e_iter))
    # # # #cross_a_emp.append(np.mean(cross_a_iter))
    # # # cross_w_emp_std.append(np.std(cross_w_iter))
    # # # cross_e_emp_std.append(np.std(cross_e_iter))
    # # #cross_a_emp_std.append(np.std(cross_a_iter))
    # # #spec_gaps_w_std.append(np.std(spec_gaps_w_iter))
    # # # spec_gaps_e_std.append(np.std(spec_gaps_e_iter))
    # # # #spec_gaps_a_std.append(np.std(spec_gaps_a_iter))
    spec_gaps_truth.append(truth_spec)
    # #spec_gaps_train.append(train_spec)




#print(hi)
index = [i*step for i in range(start,k)]
plt.plot(index,spec_gaps_w,label='W State')
plt.plot(index,spec_gaps_e,label='Stateless')
#plt.plot(range(1,k),spec_gaps_w_raw,label='Spec Gap Random Walk Raw')
#plt.plot(index,spec_gaps_a,label='FMM')
#plt.plot(range(1,k),spec_gaps_a,label='Spec Gap Combo Mixed')
#plt.plot(range(1,k),spec_gaps_e_raw,label='Spec Gap Combo Raw')
# # #plt.errorbar(range(1,5),spec_gaps_a,yerr=spec_gaps_a_std,label='Spec Gap Train w/ Path Length 2 temp change edges missing')
plt.plot(index,spec_gaps_truth,label='Spec Gap Truth')
plt.xlabel('Num Training Iterations')
plt.ylabel('Spectral Gap of Generated Expectation Matrix')
#plt.plot(range(1,k),spec_gaps_train,label='Spec Gap Train')
plt.legend(loc='best')
plt.savefig('plots/three_cluster_line__specGaps_expected_walk_statecomp.pdf')
plt.gcf().clear()

#print(hi)
#plt.plot(range(1,k),spec_gaps_w_emp,label='Spec Gap 2 Iter')
#plt.plot(range(1,k),spec_gaps_e_emp,label='Spec Gap Stateless')
#plt.plot(range(1,k),spec_gaps_w_raw,label='Spec Gap Random Walk Raw')
#plt.plot(range(1,k),spec_gaps_a,label='Spec Gap Combo Mixed')
#plt.plot(range(1,k),spec_gaps_a_emp,label='Spec Gap Combo Mixed')
#plt.plot(range(1,k),spec_gaps_e_raw,label='Spec Gap Combo Raw')
# # #plt.errorbar(range(1,5),spec_gaps_a,yerr=spec_gaps_a_std,label='Spec Gap Train w/ Path Length 2 temp change edges missing')
#plt.plot(range(1,k),spec_gaps_truth,label='Spec Gap Truth')
#plt.plot(range(1,k),spec_gaps_train,label='Spec Gap Train')
#plt.legend(loc='best')
#plt.savefig('plots/three_cluster_line_specGaps_emp_comp.pdf')
#plt.gcf().clear()
plt.plot(index,aucs_w_valid,label='W State')
plt.plot(index,aucs_e_valid,label='Stateless')
plt.legend(loc='best')
plt.xlabel('Num Training Iterations')
plt.ylabel('ROC AUC Score')
plt.savefig('plots/three_cluster_line_auc_valid_expected_walk_statecomp..pdf')
plt.gcf().clear()

plt.plot(index,aucs_w_test,label='W State')
plt.plot(index,aucs_e_test,label='Stateless')
plt.legend(loc='best')
plt.xlabel('Num Training Iterations')
plt.ylabel('ROC AUC Score')
plt.savefig('plots/three_cluster_line_auc_test_expected_walk_statecomp..pdf')
plt.gcf().clear()

plt.plot(index,aps_w_valid,label='W State')
#plt.plot(index,aps_a,label='FMM')
plt.plot(index,aps_e_valid,label='Stateless')
plt.legend(loc='best')
plt.xlabel('Num Training Iterations')
plt.ylabel('Average Precision Score')
plt.savefig('plots/three_cluster_line_edge_prediction_valid_expected_walk_statecomp..pdf')
plt.gcf().clear()

plt.plot(index,aps_w_test,label='W State')
#plt.plot(index,aps_a,label='FMM')
plt.plot(index,aps_e_test,label='Stateless')
plt.legend(loc='best')
plt.xlabel('Num Training Iterations')
plt.ylabel('Average Precision Score')
plt.savefig('plots/three_cluster_line_edge_prediction_test_expected_walk_statecomp..pdf')
plt.gcf().clear()
#print(hi)

plt.plot(index,num_edges_crossing,label='W State From 1 to 2')
plt.plot(index,num_edges_crossing_e,label='Stateless From 1 to 2')
plt.plot(index,num_edges_crossing_2,label='W State Clusters 2 to 3')
plt.plot(index,num_edges_crossing_e_2,label='Stateless Clusters 2 to 3')
plt.plot(index,num_edges_crossing_3,label='W State Clusters 1 to 3')
plt.plot(index,num_edges_crossing_e_3,label='Stateless Clusters 1 to 3')
plt.legend(loc='best')
plt.xlabel('Num Training Iterations')
plt.ylabel('Num Edges Crossing Between Clusters')
plt.savefig('plots/three_cluster_line__edges_crossing_expected_walk_statecomp..pdf')
plt.gcf().clear()

# plt.plot(index,num_edges_crossing,label='Clusters 1 to 2')
# plt.plot(index,num_edges_crossing_2,label='Clusters 2 to 3')
# plt.plot(index,num_edges_crossing_3,label='Clusters 1 to 3')
# plt.legend(loc='best')
# plt.xlabel('Num Training Iterations')
# plt.ylabel('Num Edges Crossing Between Clusters')
# plt.savefig('plots/three_cluster_line_walk_edges_crossing_wstate_disc2iter.pdf')
# plt.gcf().clear()

# plt.plot(index,num_edges_crossing_e,label='Clusters 1 to 2')
# plt.plot(index,num_edges_crossing_e_2,label='Clusters 2 to 3')
# plt.plot(index,num_edges_crossing_e_3,label='Clusters 1 to 3')
# plt.legend(loc='best')
# plt.xlabel('Num Training Iterations')
# plt.ylabel('Num Edges Crossing Between Clusters')
# plt.savefig('plots/three_cluster_line_walk_edges_crossing_wstate_disc1iter.pdf')
# plt.gcf().clear()



#plt.errorbar(range(1,k),cross_w_emp,yerr = cross_w_emp_std, label='Num Edges Crossing 2 Iter')
#plt.errorbar(range(1,k),cross_e_emp,yerr = cross_e_emp_std, label='Num Edges Crossing Stateless')
#plt.plot(range(1,k),num_edges_crossing_raw,label='Num Edges Crossing Random Walk Raw')
#plt.plot(range(1,k),num_edges_crossing_e_raw,label='Num Edges Crossing Combo Raw')
#plt.errorbar(range(1,k),cross_a_emp,yerr = cross_a_emp_std, label='Num Edges Crossing Combo Mixed')
#plt.plot(range(1,k),num_edges_crossing_a_raw,label='Num Edges Crossing Mixed Raw')
#plt.legend(loc='best')
#plt.savefig('plots/three_cluster_line_stateless_keephidden/three_cluster_line_edges_crossing_.pdf')
#plt.gcf().clear()
plt.plot(index,three_cluster_line_edge_density_w,label='W State Connect Edge Cluster 1 to 2')
plt.plot(index,three_cluster_line_edge_density_e,label='Stateless Connect Edge Cluster 1 to 2')
plt.plot(index,three_cluster_line_edge_density_w_2,label='W State Connect Edge Cluster 2 to 3')
plt.plot(index,three_cluster_line_edge_density_e_2,label='Stateless Connect Edge Cluster 2 to 3')
#plt.plot(index,three_cluster_line_edge_density_a,label='FMM')
plt.legend(loc='best')
plt.xlabel('Num Training Iterations')
plt.ylabel('Expected Value Target Crossing Edge')
plt.savefig('plots/three_cluster_line_edge_density_expected_walk_statecomp.pdf')
plt.gcf().clear()
print(hi)
# plt.gcf().clear()
# print(eo_means)
# print(em_means)
# print(eo_means_f)
# plt.gcf().clear()
plt.errorbar(range(1,k),eo_means,yerr=eo_stds,label='Fraction Training Edges Generated')
plt.errorbar(range(1,k),em_means,yerr=em_stds,label='Fraction Training Non-Edges Omitted')
plt.errorbar(range(1,k),eo_means_f,yerr=eo_stds_f,label='Fraction Ground-Truth Edges Generated')
plt.errorbar(range(1,k),em_means_f,yerr=em_stds_f,label='Fraction Ground-Truth Non-Edges Omitted')
#plt.plot(range(1,),spec_gaps_truth,label='Spec Gap Truth')
# plt.savefig('plots/three_cluster_line_specGaps_em_training.pdf')
# plt.gcf().clear()
plt.legend(loc='best')
#plt.show()
plt.savefig('plots/three_cluster_line_edge_edgeDiscs_expected_walk_statecomp..pdf')
plt.gcf().clear()
plt.plot(range(1,k),aucs,label='AUC Scores')
plt.plot(range(1,k),aps,label='Edge Precision Scores')
plt.legend(loc='best')
plt.savefig('plots/three_cluster_line_edge_prediction.pdf')
plt.gcf().clear()
plt.errorbar(range(1,k),es_means,yerr=es_stds,label='Average Score Edges')
plt.errorbar(range(1,k),nes_means,yerr=nes_stds,label='Average Score NonEdges')
plt.errorbar(range(1,k),es_v_means,yerr=es_v_stds,label='Average Score Edges Validation')
plt.errorbar(range(1,k),nes_v_means,yerr=nes_v_stds,label='Average Score NonEdges Validation')
#plt.show()
plt.savefig('plots/three_cluster_line_edge_edgeDiscsExpected_stateComp_disc1_iter.pdf')


# train_error_edge = []
# train_error_walk = []
# for i in range(N):
#     for j in range(N):
#         train_error_edge.append(A_matrix[i][j]-X_e[i][j])
#         train_error_walk.append(A_matrix[i][j]-X_w[i][j])
# print(max(train_error_edge))
# print(max(train_error_walk))
# bins = np.arange(0,max(max(train_error_edge),max(train_error_walk)),.005)
# plt.hist(train_error_walk, bins, alpha=0.5, label='walk')
# plt.hist(train_error_edge, bins, alpha=0.5, label='edge')
# plt.legend(loc='upper right')
# plt.show()


# expected_graph = np.loadtxt('plots/three_cluster_line_rw/trainingIteration_7000_expectedGraph.txt')
# X = genExpected_fromWalks(expected_graph,A_matrix.sum())

# #walk_emp = np.loadtxt('../three_cluster_line_stateless_keephiddenp.txt')
# # combo_emp = np.loadtxt('../three_cluster_line_stateless_keephidden_emp.txt')
# # combo_exp = genExpected_fromWalks(combo_emp,A_matrix.sum())

# # diff = A_matrix-combo_exp
# # diffs_s, l2_s, colors = againstFiedler(diff,u,A_matrix,1,0)
# # plt.scatter(diffs_s,l2_s,s=10, c=colors)
# # plt.savefig('plots/three_cluster_line_stateless_keephidden/combo_emp_10_fiedler_plots.pdf')
# # plt.gcf().clear()
# # fig, ax = plt.subplots()
# # norm = MidpointNormalize(midpoint=0)
# # im = ax.imshow(diff, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
# # fig.colorbar(im)
# # plt.savefig('plots/three_cluster_line_stateless_keephidden/diff_10_heat.pdf')

# #sampled_graph = np.loadtxt('plots/three_cluster_line_rw/trainingIteration_19500_randGraph.txt')
# sampled_graph = gnp(X)

# # np.savetxt('plots/19500_gnp_adj_nz.txt',sampled_graph)
# #norm = MidpointNormalize(midpoint=0)
# #fig, ax = plt.subplots()
# #im = ax.imshow(expected_graph, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
# #fig.colorbar(im)
# # plt.savefig('plots/19500_gnp_adj_nz_heat.pdf')
# # norm = MidpointNormalize(midpoint=0)
# # fig, ax = plt.subplots()
# # im = ax.imshow(X, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
# # fig.colorbar(im)
# # plt.savefig('plots/19500_expected_nz_adj_heat.pdf')

# stats = utils.compute_graph_statistics(sampled_graph)


# sps_truth = utils.sp(A_matrix)
# sps_gen = utils.sp(sampled_graph)

# stats['sp_emd']= utils.emd(sps_truth,sps_gen)

# stats['spec_gap_l2']=(utils.specGap(A_matrix)-utils.specGap(sampled_graph))**2
# stats['spec_gap_l1']=np.abs(utils.specGap(A_matrix)-utils.specGap(sampled_graph))
# stats['spec_gap_raw']=utils.specGap(A_matrix)-utils.specGap(sampled_graph)
# f = open('plots/three_cluster_line_rw/7000_gnp_stats.txt',"w")
# f.write( str(stats) )
# f.close()
# # P_from_scores, mix_from_scores_slem = utils.fast_mix_transition_matrix_slem(sp.csr_matrix(expected_graph))
# # print(mix_from_scores_slem)
# # P_from_scores, mix_from_scores_slem = utils.fast_mix_transition_matrix_slem(sp.csr_matrix(X))
# # print(mix_from_scores_slem)
# # P_from_scores, mix_from_scores_slem = utils.fast_mix_transition_matrix_slem(sp.csr_matrix(sampled_graph))
# # print(mix_from_scores_slem)





