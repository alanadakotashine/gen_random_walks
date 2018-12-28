
# coding: utf-8

# In[1]:

import sys 
import faulthandler 

faulthandler.enable()

sys.path.append('../')
sys.path.insert(0, 'netgan/')

for p in sys.path:
    print(p)



from netgan import *
import tensorflow as tf
from netgan import utils
import scipy.sparse as sp
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
import time
from matplotlib.colors import Normalize

import random



class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def walk_generator(walker,walk_type):
    if walk_type == 'walk':
        return walker.walk
    else:
        return walker.combo_walk



#get_ipython().magic(u'matplotlib inline')


# #### Load the data

# In[2]:

if __name__ == "__main__":
    random.seed()
    netgan_seed = random.randint(1,100)
    num_disc_iters = sys.argv[1]
    walk_type = sys.argv[2]
    wstate = sys.argv[3]

    netgan_seed = str(netgan_seed)
    params = [netgan_seed, num_disc_iters, walk_type, wstate]
    with open('netgan/plots/netgan_params.txt', 'w') as f:
        for item in params:
            f.write("%s\n" % item)
    f.close()

    num_disc_iters = int(num_disc_iters)
    wstate = int(wstate)
    netgan_seed = int(netgan_seed)




    _A_obs, _X_obs, _z_obs = utils.load_npz('data/cora_ml.npz')
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    lcc = utils.largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc,:][:,lcc]
    _N = _A_obs.shape[0]

    A = _A_obs.todense()
    # clusters = []
    # G = nx.Graph()
    # for i in range(5):
    #     H = nx.gnp_random_graph(20,.7)
    #     H = nx.relabel_nodes(H,dict(zip(range(20),range(i*20,(i+1)*20))))
    #     G = nx.union(H,G)
    # print(G.nodes())
    #0-19
    #20-39
    #40-59
    #60-79
    #70-99
    # G.add_edge(15,25)
    # G.add_edge(35,46)
    # G.add_edge(48,62)
    # G.add_edge(68,95)
    # _A_obs = nx.adjacency_matrix(G)
    # A = _A_obs.todense()
    #A = np.loadtxt('netgan/plots/cora_ml.txt')

    # print(A[25,50])
    # print(A[55,125])
    # print(A[127,185])
    #print(nx.is_connected(G))
    #G = nx.karate_club_graph()
    #G = nx.read_gml('data/mayan_gc.gml')
    #A = nx.adjacency_matrix(G).todense()
    #L = nx.normalized_laplacian_matrix(G).todense()
    #w,v = np.linalg.eig(L)
    #w = sorted(w)
    #A = np.loadtxt('netgan/plots/cora_ml.txt')
    #G = nx.from_numpy_matrix(A)

    #H = nx.gnp_random_graph(50,.9)
    #H = nx.relabel_nodes(H,dict(zip(range(50),range(100,150))))
    #G=nx.union(H,G)
    #G.add_edge(80,130)











    #A_matrix= np.load('data/tri_clust.txt.npy')
    #_A_obs = sp.csr_matrix(A_matrix)


    #A = np.array([[0,1,0,0],[1,0,1,1],[0,1,0,1],[0,1,1,0]])
    #_A_obs = sp.csr_matrix(A)

    #G = nx.grid_graph([10,10])
    #mapping = dict(zip(sorted(G.nodes()),range(100)))
    #G = nx.relabel_nodes(G,mapping)
    #A = nx.adjacency_matrix(G).todense()

    num_edges = A.sum()


    _A_obs = sp.csr_matrix(A)
    print(_A_obs.shape)

    #print(_A_obs)
    _A_obs[_A_obs > 1] = 1
    lcc = utils.largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc,:][:,lcc]
    _N = _A_obs.shape[0]








    # In[3]:


    val_share = .1
    test_share = 0.05
    seed = 481516234


    # #### Separate the edges into train, test, validation

    # In[4]:
    print(num_edges)
    print(type(_A_obs))


    train_ones, val_ones, val_zeros, test_ones, test_zeros = utils.train_val_test_split_adjacency(_A_obs, val_share, test_share, seed, undirected=True, connected=True, asserts=True, set_ops=False)

    train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1]))).tocsr()
    assert (train_graph.toarray() == train_graph.toarray().T).all()
    train_graph_a = train_graph.todense()
    print(train_graph_a.shape)
    #print(train_graph_a[80,130])
    #print(train_graph_a[66,99])
    np.savez('netgan/plots/cora_ml',A)
    np.savez('netgan/plots/cora_ml_val_edges',val_ones)
    np.savez('netgan/plots/cora_ml_val_non_edges',val_zeros)
    np.savez('netgan/plots/cora_ml_test_edges',test_ones)
    np.savez('netgan/plots/cora_ml_test_non_edges',test_zeros)
    np.savez('netgan/plots/cora_ml_training',train_graph_a)





    # degree_train = train_graph_a.sum(axis=0).tolist()[0]


    # E = np.loadtxt('netgan/plots/barbell_edge_wd_em/trainingIteration_2000_expectedGraph.txt')
    # E_W = np.loadtxt('netgan/plots/barbell_walk_ep/trainingIteration_1900_expectedGraph.txt')
    # vals = []
    # vals_W = []
    # vals_degree = []
    # vals_z = []
    # vals_W_z = []
    # truth_train = []
    # for [i,j] in train_ones.tolist():
    #     print(A[i,j])
    # for [i,j] in val_ones.tolist():
    #     vals.append(E[i][j])
    #     vals_W.append(E_W[i][j])
    #     print(degree_train)
    #     ds = [degree_train[i],degree_train[j]]
    #     print(ds)
    #     vals_degree.append(np.mean(ds))
    # for [i,j] in val_zeros.tolist():
    #     vals_z.append(E[i][j])
    #     vals_W_z.append(E_W[i][j])
    # # bins = np.arange(0,1+.005,.005)
    # # plt.hist(vals, bins, alpha=0.5, label='edge_one')
    # # plt.hist(vals_W, bins, alpha=0.5, label='walk_one')
    # # plt.hist(vals_z, bins, alpha=0.5, label='edge_zero')
    # # plt.hist(vals_W_z, bins, alpha=0.5, label='walk_zero')
    # # plt.legend(loc='upper right')
    # # plt.show()
    # plt.scatter(vals,vals_degree,label='edge train one')
    # plt.scatter(vals_W,vals_degree,label='walk train one')
    # plt.scatter(vals_z,vals_degree,label='edge train zero')
    # plt.scatter(vals_W_z,vals_degree,label='walk train zero')
    # plt.legend(loc='lower right')
    # plt.show()
    # print(hi)

    # #print(test_ones)

    # # In[5]:


    # train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1]))).tocsr()
    # assert (train_graph.toarray() == train_graph.toarray().T).all()


    # #### Parameters

    # In[6]:

    #length of the walk
    rw_len = 16
    #batch_size is how many data-points are fed into the discriminator
    batch_size = 12*15
    #number of nodes in the walk
    data_size = 16

    #train_graph = _A_obs.tocoo()
    #val_ones = np.empty(np.shape(val_ones))
    #val_zeros = np.empty(np.shape(val_zeros))



    # In[7]:



    walker = utils.RandomWalker(train_graph, rw_len, p=1, q=1, batch_size=batch_size)
    #print(hi)
    batch_size = 12*15
    M = walker.RW
    walk_gen = walk_generator(walker,walk_type)

    
    # np.savetxt('netgan/plots/barbell_transition_simpleWalk_sanity.txt',M)
    #RW_ex = walker.RW_ex_correct
    # fmm_ex = walker.fmm_ex
    #np.savetxt('netgan/plots/barbell_rw_expected_correct.txt',RW_ex)
    # np.savetxt('netgan/plots/barbell_fmm_expected.txt',fmm_ex)
    #RW_entropy = utils.graph_entropy_matrix(RW_ex)
    # print(RW_entropy.sum())
    # fmm_entropy = utils.graph_entropy_matrix(fmm_ex)
    # print(fmm_entropy.sum())

    #np.savetxt('netgan/plots/barbell_rw_entropy_correct.txt',RW_entropy)

    # np.savetxt('netgan/plots/barbell_fmm_entropy.txt',fmm_entropy)


    # P = walker.P 

    x=walker.walk().next()
    edges = []
    for walk in x:
       for i in range(15):
           edges.append([walk[i],walk[i+1]])
    edge_data = np.zeros([_N,_N])
    for [i,j] in edges:
        edge_data[i][j] = edge_data[i][j]+1
        edge_data[j][i] = edge_data[j][i]+1
    scale = float(num_edges)/float(edge_data.sum())
    edge_data_nz = np.copy(edge_data)
    for i in range(_N):
        edge_data_nz[i][i]=0
    edge_data = edge_data*scale
    scale = float(num_edges)/float(edge_data_nz.sum())
    edge_data_nz = edge_data_nz*scale
    fig, ax = plt.subplots()
    norm = MidpointNormalize(midpoint=0)
    im = ax.imshow(edge_data_nz, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    fig.colorbar(im)
    plt.savefig('netgan/plots/cora_ml_walk_emp_heat.pdf')
    plt.gcf().clear()
    #np.savetxt('netgan/plots/cora_ml_walk_emp.txt',edge_data_nz)

    x=walker.fast_mix_walk().next()
    edges = []
    for walk in x:
        for i in range(15):
            edges.append([walk[i],walk[i+1]])
    print(len(edges))
    edge_data = np.zeros([_N,_N])
    for [i,j] in edges:
        edge_data[i][j] = edge_data[i][j]+1
        edge_data[j][i] = edge_data[j][i]+1
    scale = float(num_edges)/float(edge_data.sum())
    edge_data_nz = np.copy(edge_data)
    for i in range(_N):
        edge_data_nz[i][i]=0
    edge_data = edge_data*scale
    scale = float(num_edges)/float(edge_data_nz.sum())
    edge_data_nz = edge_data_nz*scale
    fig, ax = plt.subplots()
    norm = MidpointNormalize(midpoint=0)
    im = ax.imshow(edge_data_nz, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    fig.colorbar(im)
    plt.savefig('netgan/plots/cora_ml_fmm_emp_heat.pdf')
    plt.gcf().clear()
    #np.savetxt('netgan/plots/cora_ml_fmm_emp.txt',edge_data_nz)


    x=walker.combo_walk().next()
    edges = []
    for walk in x:
        for i in range(15):
            edges.append([walk[i],walk[i+1]])
    edge_data = np.zeros([_N,_N])
    for [i,j] in edges:
        edge_data[i][j] = edge_data[i][j]+1
        edge_data[j][i] = edge_data[j][i]+1
    scale = float(num_edges)/float(edge_data.sum())
    edge_data_nz = np.copy(edge_data)
    for i in range(_N):
        edge_data_nz[i][i]=0
    edge_data = edge_data*scale
    scale = float(num_edges)/float(edge_data_nz.sum())
    edge_data_nz = edge_data_nz*scale
    fig, ax = plt.subplots()
    norm = MidpointNormalize(midpoint=0)
    im = ax.imshow(edge_data_nz, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    fig.colorbar(im)
    plt.savefig('netgan/plots/cora_ml_combo_walk5_fmm5_emp_heat.pdf')
    plt.gcf().clear()
    #np.savetxt('netgan/plots/cora_ml_combo_walk5_fmm5_emp.txt',edge_data_nz)



    # print(hi)
    # s = walker.fmm_time 
    # np.savetxt('netgan/plots/barbell_transition_fmm_time.txt',[s])

    # np.savetxt('netgan/plots/barbell_transition_fmm.txt',P)

    # diff_transition = P-M

    # fig, ax = plt.subplots()
    # norm = MidpointNormalize(midpoint=0)
    # im = ax.imshow(diff_transition, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    # fig.colorbar(im)
    # plt.savefig('barbell_transition_diff_heat.pdf')
    # plt.gcf().clear()

    # fig, ax = plt.subplots()
    # norm = MidpointNormalize(midpoint=0)
    # im = ax.imshow(M, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    # fig.colorbar(im)
    # plt.savefig('barbell_transition_simpleWalk_heat.pdf')
    # plt.gcf().clear()
    # fig, ax = plt.subplots()
    # norm = MidpointNormalize(midpoint=0)
    # im = ax.imshow(P, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    # fig.colorbar(im)
    # plt.savefig('barbell_transition_fmmWalk_heat.pdf')
    # plt.gcf().clear()
    # #plt.imshow(edge_data, cmap='ho







    # # #### An example random walk

    # # In[8]:

    # x=walker.edge().next()
    # edges = []
    # for walk in x:
    #     for i in range(1):
    #         edges.append([walk[i],walk[i+1]])
    # edge_data = np.zeros([_N,_N])
    # for [i,j] in edges:
    #     edge_data[i][j] = edge_data[i][j]+1
    #     edge_data[j][i] = edge_data[j][i]+1
    # scale = float(num_edges)/float(edge_data.sum())
    # edge_data_nz = np.copy(edge_data)
    # for i in range(_N):
    #     edge_data_nz[i][i]=0
    # edge_data = edge_data*scale
    # scale = float(num_edges)/float(edge_data_nz.sum())
    # edge_data_nz = edge_data_nz*scale
    # np.savetxt('barbell_walk_emp.txt',edge_data_nz)

    # #edge_data[edge_data>1.0]=1.0
    # # print('walk')
    # # print(edge_data[np.ix_(range(50),range(50))].sum())
    # # print(edge_data[np.ix_(range(50),range(50,100))].sum())
    # # print(edge_data[np.ix_(range(50),range(100,150))].sum())
    # # print(edge_data[np.ix_(range(50,100),range(50,100))].sum())
    # # print(edge_data[np.ix_(range(50,100),range(100,150))].sum())
    # # print(edge_data[np.ix_(range(100,150),range(100,150))].sum())
    # fig, ax = plt.subplots()
    # norm = MidpointNormalize(midpoint=0)
    # im = ax.imshow(edge_data, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    # fig.colorbar(im)
    # #plt.imshow(edge_data, cmap='hot', interpolation='nearest')
    # plt.savefig('barbell_walk.pdf')
    # plt.gcf().clear()
    # fig, ax = plt.subplots()
    # edge_data[edge_data>1.0]=1.0
    # edge_data[edge_data<.00000000000001]=0
    # norm = MidpointNormalize(midpoint=0)
    # im = ax.imshow(edge_data-A, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    # fig.colorbar(im)
    # #plt.imshow(edge_data, cmap='hot', interpolation='nearest')
    # plt.savefig('barbell_walk_disc.pdf')
    # plt.gcf().clear()

    # x=walker.fast_mix_walk().next()
    # edges = []
    # for walk in x:
    #     for i in range(15):
    #         edges.append([walk[i],walk[i+1]])
    # edge_data2 = np.zeros([_N,_N])
    # for [i,j] in edges:
    #     edge_data2[i][j] = edge_data2[i][j]+1
    #     edge_data2[j][i] = edge_data2[j][i]+1
    # edge_data2_nz = np.copy(edge_data2)
    # for i in range(_N):
    #     edge_data2_nz[i][i]=0
    # scale = float(num_edges)/float(edge_data2.sum())
    # edge_data2 = edge_data2*scale
    # scale = float(num_edges)/float(edge_data2_nz.sum())
    # edge_data2_nz = edge_data2_nz*scale
    # np.savetxt('barbell_fmm_emp.txt',edge_data2_nz)
    # #edge_data[edge_data>1.0]=1.0
    # # print('fmm_walk')
    # # print(edge_data[np.ix_(range(50),range(50))].sum())
    # # print(edge_data[np.ix_(range(50),range(50,100))].sum())
    # # print(edge_data[np.ix_(range(50),range(100,150))].sum())
    # # print(edge_data[np.ix_(range(50,100),range(50,100))].sum())
    # # print(edge_data[np.ix_(range(50,100),range(100,150))].sum())
    # # print(edge_data[np.ix_(range(100,150),range(100,150))].sum())
    # norm = MidpointNormalize(midpoint=0)
    # im = ax.imshow(edge_data2, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    # fig.colorbar(im)
    # #plt.imshow(edge_data, cmap='hot', interpolation='nearest')
    # plt.savefig('barbell_fmm_walk.pdf')
    # plt.gcf().clear()
    # fig, ax = plt.subplots()
    # edge_data2[edge_data2>1.0]=1.0
    # edge_data2[edge_data2<.00000000000001]=0
    # norm = MidpointNormalize(midpoint=0)
    # im = ax.imshow(edge_data2-A, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    # fig.colorbar(im)
    # #plt.imshow(edge_data, cmap='hot', interpolation='nearest')
    # plt.savefig('barbell_fmm_walk_disc.pdf')
    # plt.gcf().clear()
    # fig, ax = plt.subplots()
    # norm = MidpointNormalize(midpoint=0)
    # im = ax.imshow(edge_data2-edge_data, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    # fig.colorbar(im)
    # #plt.imshow(edge_data, cmap='hot', interpolation='nearest')
    # plt.savefig('barbell_fmm_rw_walk_disc.pdf')
    # plt.gcf().clear()
    # fig, ax = plt.subplots()
    # norm = MidpointNormalize(midpoint=0)
    # im = ax.imshow(edge_data2_nz-edge_data_nz, norm=norm, cmap=plt.cm.seismic, interpolation='nearest')
    # fig.colorbar(im)
    # #plt.imshow(edge_data, cmap='hot', interpolation='nearest')
    # plt.savefig('barbell_fmm_rw_walk_disc_nz.pdf')
    # plt.gcf().clear()




    # ##s = walker.fmm_time
    # #np.savetxt('netgan/plots/truth_mix_time.txt',s)





    # # #### Create our NetGAN model

    # # In[9]:

    



    netgan = NetGAN(_N, data_size, walk_gen, gpu_id=0, use_gumbel=True, disc_iters=num_disc_iters,
                    W_down_discriminator_size=128, W_down_generator_size=128,
                    l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5,
                    generator_layers=[40], discriminator_layers=[30], temp_start=5, 
                    learning_rate=0.0003, batch_size=batch_size, wasserstein_penalty=10, seed=netgan_seed, wstate = wstate)


    # #### Define the stopping criterion

    # In[10]:


    stopping_criterion = "eo"

    assert stopping_criterion in ["val", "eo"], "Please set the desired stopping criterion."

    if stopping_criterion == "val": # use val criterion for early stopping
        stopping = None
    elif stopping_criterion == "eo":  #use eo criterion for early stopping
        stopping = .95# set the target edge overlap here


    # #### Train the model

    # In[12]:


    eval_every = 1000
    plot_every = 1000


    # In[ ]:


    log_dict = netgan.train(A_orig=_A_obs, val_ones=val_ones, val_zeros=val_zeros, stopping=stopping,
                            eval_every=eval_every, plot_every=plot_every, max_patience=2, max_iters=20001)


    # In[14]:


    log_dict.keys()


    # In[15]:

    print('finished training')


    plt.plot(np.arange(len(log_dict['val_performances'])) * eval_every, 
            np.array(log_dict['val_performances'])[:,0], label="ROC-AUC")
    plt.plot(np.arange(len(log_dict['val_performances'])) * eval_every,
            np.array(log_dict['val_performances'])[:,1], label="Avg. Prec.")

    plt.title("Validation performance during training")
    plt.legend()
    plt.savefig('netgan/plots/valid_perform.pdf')
    np.savetxt('netgan/plots/valid_perform_data.txt',log_dict['val_performances'])
    plt.gcf().clear()


    # In[16]:
    print(np.arange(len(np.array(log_dict['edge_overlaps']))))
    print(np.array(log_dict['edge_overlaps'])/float(_A_obs.sum()))
    print(_A_obs.sum())
    plt.plot(np.arange(len(np.array(log_dict['edge_overlaps']))), np.array(log_dict['edge_overlaps'])/float(_A_obs.sum()))
    plt.title("Edge overlap during training")
    plt.savefig('netgan/plots/eo.pdf')
    plt.gcf().clear()

    plt.plot(np.arange(len(np.array(log_dict['max_discrepancy']))), np.array(log_dict['max_discrepancy']))
    plt.title("Maximum discrepancy between exepected edge density and observed")
    plt.savefig('netgan/plots/edge_disc.pdf')
    plt.gcf().clear()

    print('finished plotting')


    # #### Generate random walks on the trained model

    # In[17]:


    sample_many = netgan.generate_discrete(1000, reuse=True)
    print(type(sample_many))
    print(sample_many[0])


    # In[18]:


    samples = []


    # In[31]:


    for _ in range(1000):
        #if (_+1) % 5 == 0:
        #    print(_)
        samples.append(sample_many.eval({netgan.tau: 0.5}))


    # #### Assemble score matrix from the random walks

    # In[32]:


    rws = np.array(samples).reshape([-1, data_size])
    scores_matrix = utils.score_matrix_from_random_walks(rws, _N).tocsr()
    scores_matrix_norm = scores_matrix.copy()
    scores_matrix_norm.data = scores_matrix_norm.data / np.sum(scores_matrix_norm.data)
    diff = np.asarray(scores_matrix_norm.todense()) - M
    np.savetxt('netgan/plots/diff_from_expected.txt',diff)
    np.savetxt('netgan/plots/scores.txt',np.asarray(scores_matrix_norm.todense()))

    transition_tensor, edges = utils.transition_tensor_from_random_walks(rws, _N)




    # #### Evaluate generalization via link prediction

    # In[33]:


    test_labels = np.concatenate((np.ones(len(test_ones)), np.zeros(len(test_zeros))))
    test_scores = np.concatenate((scores_matrix[tuple(test_ones.T)].A1, scores_matrix[tuple(test_zeros.T)].A1))


    # In[37]:


    print(roc_auc_score(test_labels, test_scores))


    # In[38]:


    print(average_precision_score(test_labels, test_scores))



    A_select = train_graph
    print(A_select.sum())
    sampled_graph = utils.graph_from_scores(scores_matrix, A_select.sum())

    np.savetxt('netgan/plots/sampled_graph.txt',sampled_graph)

    stats = utils.compute_graph_statistics(sampled_graph)
    f = open('netgan/plots/stats.txt',"w")
    f.write( str(stats) )
    f.close()

    sampled_graph_from_walk = utils.graph_from_transitions(transition_tensor, edges, A_select.sum(), _N)
    print(type(sampled_graph_from_walk))
    print(type(sampled_graph))
    np.savetxt('netgan/plots/sampled_graph_from_walk.txt',sampled_graph_from_walk)
    stats = utils.compute_graph_statistics(sampled_graph_from_walk)
    f = open('netgan/plots/stats_from_walk.txt',"w")
    f.write( str(stats) )
    f.close()




    #truth = utils.compute_graph_statistics(A_matrix)
    #f = open('netgan/plots/truth.txt',"w")
    #f.write( str(truth) )
    #f.close()






