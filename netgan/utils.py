import networkx as nx
import scipy.sparse as sp2
import numpy as np
import numpy.linalg as linalg
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.stats as spst
import warnings
from matplotlib import pyplot as plt
import igraph
import powerlaw
import picos as pic
import cvxopt as cvx
import itertools
from numba import jit
from copy import copy

######SLEM CODE##############
def tp_rw_matrix ( graph ) :
    edges = list(graph . edges ())
    n = graph . number_of_nodes ()
    P = np . zeros (( n , n ) )
    for i , j in edges :
        d_i = len ( [n for n in graph . neighbors ( i )] )
        d_j = len ( [n for n in graph . neighbors ( j )] )
        P [i , j ] = 1./ d_i
        P [j , i ] = 1./ d_j
    return P

def tp_rw_matrix_weighted ( graph, W) :
    edges = list(graph . edges ())
    n = graph . number_of_nodes ()
    P = np . zeros (( n , n ) )
    for i , j in edges :
        d_i = W[i].sum()
        d_j = W[j].sum()
        P [i , j ] = W[i,j]/float(d_i)
        P [j , i ] = W[j,i]/float(d_j)
    return P

def mh_chain(graph):
    n = graph . number_of_nodes ()
    pi_vec = 1./ n * np . ones ( n )
    edges = graph . edges ()
    R = np . zeros (( n , n ) )
    P = np . zeros (( n , n ) )
    P_rw = tp_rw_matrix ( graph )
    for i in range ( n ) :
        for j in range ( n ) :
            R [i , j ] = ( pi_vec [ j ]* P_rw [j , i ])/( pi_vec [ i ]* P_rw [i , j ])
    for i , j in edges :
        P [i , j ] = P_rw [i , j ]* min (1 , R [i , j ])
        P [j , i ] = P [i , j ]
    for i in range ( n ) :
        s = 0
        for k in graph . neighbors ( i ) :
            s += P_rw [i , k ]*(1 - min (1 , R [i , k]) )
        P [i , i ] = P_rw [i , i ] + s
    return P

def mh_chain_weighted(graph, W):
    n = graph . number_of_nodes ()
    pi_vec = 1./ n * np . ones ( n )
    edges = graph . edges ()
    R = np . zeros (( n , n ) )
    P = np . zeros (( n , n ) )
    P_rw = tp_rw_matrix_weighted( graph, W )
    for i , j in edges :
        P [i , j ] = min(P_rw[i,j],P_rw[j,i])
        P [j, i] = P[i,j]
    for i in range ( 1,n ) :
        s = 0
        for k in graph . neighbors ( i ) :
            s += max(0,P_rw[i,k]-P_rw[k,i])
        P [i , i ] = P_rw [i , i ] + s
    return P

def slem (P) :
    eig_vals , eig_vecs = linalg.eig(P)
    eig_vals = list(eig_vals)
    eig_vals.sort()
    return max( -eig_vals [0], eig_vals [ -2]).real

def f(graph,p):
    mu = slem (tp_matrix (graph,p))
    return mu

def tp_matrix ( graph , p ) :
    edges = list(graph . edges ())
    n = graph . number_of_nodes ()
    P = np . identity ( n )
    for l in xrange (len ( edges ) ) :
        E = np . zeros ([ n , n ] , dtype = float )
        i , j = edges [ l ]
        E [i , j ] = 1
        E [j , i ] = 1
        E [i , i ] = -1
        E [j , j ] = -1
        P += p [ l ]* E
    return P

def sub ( graph , p ) :
    edges = list(graph . edges ())
    nodes = graph . nodes ()
    m = graph . number_of_edges ()
    n = graph . number_of_nodes ()
    P = tp_matrix ( graph , p )
    g = np . zeros ( m )
    eig_vals , eig_vecs = linalg. eig ( P )
    eig_list = zip( eig_vals , np . transpose ( eig_vecs ) )
    eig_list . sort ( key = lambda x : x [0])
    lambda_2 , lambda_n = eig_list [ -2][0] , eig_list [0][0]
    print('computed subgradient')
    print(max((lambda_2,-lambda_n)))
    if lambda_2 >= - lambda_n :
        #print('second')
        u = [ u_i . real for u_i in eig_list [ -2][1]]
        for l in xrange ( m ) :
            i , j = edges [ l ]
            g [ l ] = -( u [ i ] - u [ j ]) **2
    else :
        #print('last')
        v = [ v_i . real for v_i in eig_list [0][1]]
        for l in xrange ( m ) :
            i , j = edges [ l ]
            g [ l ] = ( v [ i ] - v [ j ]) **2
    return g

def solve (G , p0 , w, max_iter =100 , alpha = lambda k : 1./( np . sqrt ( k ) ) ):
    #global p , graph
    #print('slem')
    p = p0
    graph = G
    edges = list(graph . edges ())
    nodes = graph . nodes ()
    n = graph . number_of_nodes ()
    m = graph . number_of_edges ()
    k = 1
    sol = {'f': f ( graph , p ) ,
        'p': copy ( p ) , 'iter' : 0 ,
        'fk': np . zeros ( max_iter +1) }
    sol ['fk'][0] = f ( graph , p )
    while k <= max_iter :
        # subgradient step
        g = sub ( graph , p )
        #print('sub')
        # sequential projection step
        p -= alpha ( k ) / linalg. norm ( g ) * g
        for l in range ( m ) : p [ l ] = max ( p [ l ] , 0)
        #p can be at most w
        for l in range (m): p[l] = min(p[l],w[l])
        for i in range ( n ) :
            I = [ l for l in xrange ( m ) if i in edges[ l ]]
            while sum ([ p [ l ] for l in I ]) > 1:
                I = [ l for l in I if p [ l ] > 0]
                p_min = min ([ p [ l ] for l in I ])
                p_sum = sum ([ p [ l ] for l in I ])
                delta = min( p_min , ( p_sum - 1.)/len( I ) )
                for l in I : p [ l ] -= delta
        sol ['fk'][ k ] = f ( graph , p )
        if f ( graph , p ) < sol ['f']:
            sol ['f'] = f ( graph , p )
            sol ['p'] = copy ( p )
            sol ['iter'] = k
        k += 1
    return sol

def graph_values_to_vector ( graph , P ) :
    edges = list(graph . edges ())
    m = graph . number_of_edges ()
    p = np . zeros ( m )
    for l in range ( m ) :
        (i,j) = edges[l]
        p [ l ] = P [i , j ]
    return p

def optimize ( graph , W, chain = mh_chain_weighted, max_iter =200 , alpha = lambda k : 1./ np . sqrt ( k ) ):
    P = chain ( graph, W )
    p = graph_values_to_vector ( graph , P )
    w = graph_values_to_vector (graph, W)
    sol = solve ( graph ,p , w, max_iter , alpha )
    return sol

def const_steplength ( h ) :
    def alpha ( k ) :
        return float ( h ) / linalg. norm ( sub ( graph , p ) )
    return alpha

###############################


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name) as loader:
        loader = dict(loader)['arr_0'].item()
        adj_matrix = sp2.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp2.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def edges_to_sparse(edges, N, values=None):
    """
    Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    Parameters
    ----------
    edges : array-like, shape [n_edges, 2]
        Edge indices
    N : int
        Number of nodes
    values : array_like, shape [n_edges]
        The values to put at the specified edge indices. Optional, default: np.ones(.)

    Returns
    -------
    A : scipy.sparse.csr.csr_matrix
        Sparse adjacency matrix

    """
    if values is None:
        values = np.ones(edges.shape[0])

    return sp2.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                   every_node=True, connected=False, undirected=False,
                                   use_edge_cover=True, set_ops=True, asserts=False):
    """
    Split the edges of the adjacency matrix into train, validation and test edges
    and randomly samples equal amount of validation and test non-edges.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse unweighted adjacency matrix
    p_val : float
        Percentage of validation edges. Default p_val=0.10
    p_test : float
        Percentage of test edges. Default p_test=0.05
    seed : int
        Seed for numpy.random. Default seed=0
    neg_mul : int
        What multiplicity of negative samples (non-edges) to have in the test/validation set
        w.r.t the number of edges, i.e. len(non-edges) = L * len(edges). Default neg_mul=1
    every_node : bool
        Make sure each node appears at least once in the train set. Default every_node=True
    connected : bool
        Make sure the training graph is still connected after the split
    undirected : bool
        Whether to make the split undirected, that is if (i, j) is in val/test set then (j, i) is there as well.
        Default undirected=False
    use_edge_cover: bool
        Whether to use (approximate) edge_cover to find the minimum set of edges that cover every node.
        Only active when every_node=True. Default use_edge_cover=True
    set_ops : bool
        Whether to use set operations to construction the test zeros. Default setwise_zeros=True
        Otherwise use a while loop.
    asserts : bool
        Unit test like checks. Default asserts=False

    Returns
    -------
    train_ones : array-like, shape [n_train, 2]
        Indices of the train edges
    val_ones : array-like, shape [n_val, 2]
        Indices of the validation edges
    val_zeros : array-like, shape [n_val, 2]
        Indices of the validation non-edges
    test_ones : array-like, shape [n_test, 2]
        Indices of the test edges
    test_zeros : array-like, shape [n_test, 2]
        Indices of the test non-edges

    """
    assert p_val + p_test > 0
    assert A.max() == 1  # no weights
    assert A.min() == 0  # no negative edges
    assert A.diagonal().sum() == 0  # no self-loops
    assert not np.any(A.sum(0).A1 + A.sum(1).A1 == 0)  # no dangling nodes

    is_undirected = (A != A.T).nnz == 0

    if undirected:
        assert is_undirected  # make sure is directed
        print(type(A))
        A = sp2.tril(A)  # consider only upper triangular
        A = A.tocsr()
        A.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(seed)
    ones_orig = np.column_stack(A.nonzero())

    
    
    E = A.nnz
    N = A.shape[0]
    s_train = int(E * (1 - p_val - p_test))

    print(ones_orig)

    #print(hi)


    idx = np.arange(N)


    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            assert connected_components(A)[0] == 1  # make sure original graph is connected
            A_hold = minimum_spanning_tree(A)
        else:
            A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = A.sum(1).A1

            if use_edge_cover:
                hold_edges = np.array(list(nx.maximal_matching(nx.DiGraph(A))))
                not_in_cover = np.array(list(set(range(N)).difference(hold_edges.flatten())))

                # makes sure the training percentage is not smaller than N/E when every_node is set to True
                min_size = hold_edges.shape[0] + len(not_in_cover)
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(min_size / E))

                d_nic = d[not_in_cover]

                hold_edges_d1 = np.column_stack((not_in_cover[d_nic > 0],
                                                 np.row_stack(map(np.random.choice,
                                                                  A[not_in_cover[d_nic > 0]].tolil().rows))))

                if np.any(d_nic == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, not_in_cover[d_nic == 0]].T.tolil().rows)),
                                                     not_in_cover[d_nic == 0]))
                    hold_edges = np.row_stack((hold_edges, hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = np.row_stack((hold_edges, hold_edges_d1))

            else:
                # makes sure the training percentage is not smaller than N/E when every_node is set to True
                if N > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(N / E))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
                                                     idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == N

            A_hold = edges_to_sparse(hold_edges, N)

        A_hold[A_hold > 1] = 1
        A_hold.eliminate_zeros()
        A_sample = A - A_hold

        s_train = s_train - A_hold.nnz
    else:
        A_sample = A
    idx_ones = np.random.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    
    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

    n_test = len(test_ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
        else:
            random_sample = np.random.randint(0, N, [int(1.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

        test_zeros = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        test_zeros = np.row_stack(test_zeros)[:n_test]
        assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, N, 2)
            if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]


    

    val_zeros = test_zeros[:s_val_zeros]
    test_zeros = test_zeros[s_val_zeros:]

    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back original A
        symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        A = A.maximum(A.T)



    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or connected_components(A_hold)[0] == 1
        assert not every_node or ((A_hold - A) > 0).sum() == 0



    return train_ones, val_ones, val_zeros, test_ones, test_zeros

def transition_tensor_from_random_walks(random_walks, N):
    random_walks = np.array(random_walks)
    trigrams = np.array(list(zip(random_walks[:, :-2], random_walks[:, 1:-1], random_walks[:, 2:])))
    T= np.zeros([N,N,N])
    for (i,j,k) in trigrams:
        T[i,j,k] = T[i,j,k]+1
    all_pairs = list(itertools.combinations_with_replacement(range(N),2))
    edges = []
    for (i,j) in all_pairs:
        total = sum(T[i,j,:])
        if total != 0:
            edges.append((i,j))
            T[i,j,:] = T[i,j,:]/float(total)
        total = sum(T[j,i,:])
        if total != 0:
            edges.append((j,i))
            T[j,i,:] = T[j,i,:]/float(total)
    return T, edges





def score_matrix_from_random_walks(random_walks, N, symmetric=True):
    """
    Compute the transition scores, i.e. how often a transition occurs, for all node pairs from
    the random walks provided.
    Parameters
    ----------
    random_walks: np.array of shape (n_walks, rw_len, N)
                  The input random walks to count the transitions in.
    N: int
       The number of nodes
    symmetric: bool, default: True
               Whether to symmetrize the resulting scores matrix.

    Returns
    -------
    scores_matrix: sparse matrix, shape (N, N)
                   Matrix whose entries (i,j) correspond to the number of times a transition from node i to j was
                   observed in the input random walks.

    """

    random_walks = np.array(random_walks)
    bigrams = np.array(list(zip(random_walks[:, :-1], random_walks[:, 1:])))
    bigrams = np.transpose(bigrams, [0, 2, 1])
    bigrams = bigrams.reshape([-1, 2])
    if symmetric:
        bigrams = np.row_stack((bigrams, bigrams[:, ::-1]))
    data = np.ones(bigrams.shape[0])
    rows = bigrams[:, 0]
    cols = bigrams[:, 1]
    print(type(data))
    print(type(rows))
    print(type(cols))
    print(N)
    #import pdb; pdb.set_trace()
    mat = sp2.coo_matrix((data,(rows,cols)),
                        shape=(N, N))
    return mat

def compNeighbors(edges, node_ixs, i):
    N = len(node_ixs)
    if i == N-1:
        nbs = edges[node_ixs[i]:,1]
    else:
        nbs = edges[node_ixs[i]:node_ixs[i+1],1]
    return nbs

def genTransitionTensor(edges, node_ixs, p=1, q=1):
    N=len(node_ixs)
    M = np.zeros([N,N,N])
    for i in range(N):
        for j in range(N):
            nbs = compNeighbors(edges, node_ixs, j)
            prev_nbs = compNeighbors(edges, node_ixs, i)
            for k in range(N):
                #computing the probability of walking from j to k conditioned on
                #just walking fom i to j
                if k not in nbs:
                    M[i,j,k] = 0
                else:
                    if j not in prev_nbs:
                        M[i,j,k]=0
                    elif i==k:
                        M[i,j,k]=1/p
                    else:
                        if k in prev_nbs:
                            M[i,j,k]=1
                        else:
                            M[i,j,k]=1/q
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i,j,:] = np.zeros(N)
            else:
                total = sum(M[i,j,:])
                if total != 0:
                    for k in range(N):
                        M[i,j,k] = M[i,j,k]/total
    return M

def randWalkMatrix(edges, node_ixs):
    N = len(node_ixs)
    M = np.zeros([N,N])
    #print('random walk matrix')
    totalEdges = 0
    for i in range(N):
        nbs = compNeighbors(edges, node_ixs, i)
        num_nbs = len(nbs)
        totalEdges = totalEdges + num_nbs
        for j in nbs:
            #prob i move from i to j
            M[j,i]=1/float(num_nbs)
        #print(sum(M[:,i]))
    return M

def randWalkMatrix_mix_time(A):
    G = nx.from_numpy_matrix(A)
    Gc = max(nx.connected_component_subgraphs(G), key=len)
    A = np.asarray(nx.adjacency_matrix(Gc).todense())
    N = A.shape[0]
    D =  A.sum(axis=0)
    M = np.zeros((N,N))
    print(type(A))
    #print('random walk matrix')
    totalEdges = 0
    for i in range(N):
        for j in range(N):
            M[j,i]=float(A[j,i])/float(D[i])
    eig_vals , eig_vecs = linalg. eig ( M )
    eig_list = zip( eig_vals , np . transpose ( eig_vecs ) )
    eig_list . sort ( key = lambda x : x [0])
    lambda_2 , lambda_n = eig_list [ -2][0] , eig_list [0][0]
    return max(lambda_2,-1*lambda_n)

def genEdgeDist_fmm(P, rwlen):
    N = np.shape(P)[0]
    for i in range(N):
        print(sum(P[i]))
    #P_ij is probability I travel from i to j
    #so rows sum to 1
    T = np.ones(N)*(float(1/float(N)))
    for i in range(rwlen-1):
        T = np.matmul(T,P)
    M = P.copy()
    for i in range(N):
        M[i,:]=M[i,:]*T[i]
    return M


def genEdgeDist(edges, node_ixs, rwlen, p=1, q=1):
    M = randWalkMatrix(edges, node_ixs)
    T = genTransitionTensor(edges, node_ixs, p, q)
    N = len(node_ixs)
    x = np.ones(N)*1/N
    #base case: rwlen is 2. So I entrywise multiply all the columns i
#by the probability i am at that column which is x(i)
    for i in range(N):
        M[:,i] = np.multiply(M[:,i],x[i])
#now for steps 2:rwlen, I need to compute the new edge matrix. I can compute the probability I transition to a vertex l conditoned on being on any of the other edges. Now, I have to add up all the previous nodes to get the new M
    for t in range(rwlen-2):
        for i in range(N):
            for j in range(N):
                #probability i move from i to j is the probability i have traversed
                #edge k to i and now move to j for all k
                pij = T[:,i,j]
                pi = M[:,i]
                M[j,i] = np.dot(T[:,i,j],M[i,:])
    return M


def random_walk(edges, node_ixs, rwlen, p=1, q=1, n_walks=1):
    N=len(node_ixs)
    
    walk = []
    prev_nbs = None
    for w in range(n_walks):
        source_node = np.random.choice(N)
        walk.append(source_node)
        for it in range(rwlen-1):
            
            if walk[-1] == N-1:
                nbs = edges[node_ixs[walk[-1]]:,1]
            else:
                nbs = edges[node_ixs[walk[-1]]:node_ixs[walk[-1]+1],1]
                
            if it == 0:
                walk.append(np.random.choice(nbs))
                prev_nbs = set(nbs)
                continue

            is_dist_1 = []
            for n in nbs:
                is_dist_1.append(int(n in set(prev_nbs)))

            is_dist_1_np = np.array(is_dist_1)
            is_dist_0 = nbs == walk[-1]
            is_dist_2 = 1 - is_dist_1_np - is_dist_0

            alpha_pq = is_dist_0 / float(p) + is_dist_1_np + is_dist_2/float(q)
            alpha_pq_norm = alpha_pq/np.sum(alpha_pq)
            rdm_num = np.random.rand()
            cumsum = np.cumsum(alpha_pq_norm)
            nxt = nbs[np.sum(1-(cumsum > rdm_num))]
            walk.append(nxt)
            prev_nbs = set(nbs)
    return np.array(walk)




def random_walk_w_matrix(edges, node_ixs, rwlen, P, n_walks, p):
    walk = []
    N = len(node_ixs)
    for t in range(n_walks):
        i = np.random.choice(range(N),p=p)
        walk.append(i)
        for k in range(rwlen-1):
            p_ = list(P[:,i])
            total = sum(p_)
            p_ = [float(x)/total for x in p_]
            p_ = [0 if x < 0 else x for x in p_]
            i = np.random.choice(range(N), p=p_)
            walk.append(i)
    return np.array(walk)

def expected_walk_w_matrix(node_ixs, rwlen, P, s):
    N = len(node_ixs)
    S = np.zeros((N,N))
    for k in range(rwlen):
        M = np.copy(P)
        for i in range(N):
            #new edge probability, multiply each column by position in 
            #print(M[:,i].sum())
            M[:,i] = M[:,i]*s[i]
            #print(M[:,i].sum())
        #new distribution
        s = np.matmul(P,s)
        S = S + M
    return S




def random_walk_w_matrix_edges(edges, node_ixs, rwlen, P, n_walks, p):
    walk = []
    N = len(node_ixs)
    for t in range(n_walks):
        i = np.random.choice(range(N), p=p)
        #walk.append(i)
        for k in range(rwlen-1):
            p_ = list(P[:,i])
            total = sum(p_)
            p_ = [float(x)/total for x in p_]
            p_ = [0 if x < 0 else x for x in p_]
            j = np.random.choice(range(N), p=p_)
            walk.append(i)
            walk.append(j)
            i=j
    return np.array(walk)

def rand_sample_helper(x,p,low,high,rand_num):
    if low < high:
        mid = int(np.floor((high-low)/2.0)+low)
        if rand_num >= sum(p[:mid+1]) and rand_num < sum(p[:mid+2]):
            result= x[mid]
        elif rand_num < sum(p[:mid+1]):
            result= rand_sample_helper(x,p,low,mid,rand_num)
        else:
            result= rand_sample_helper(x,p,mid+1,high,rand_num)
    else:
        result= x[low]
    return result

def rand_sample_helper_slow(x,p,rand_num):
    n = len(x)
    for i in range(n):
        if rand_num <= sum(p[:i+1]):
            return x[i]
    return x[n-1]


def rand_sample(x,p):
    np.random.seed()
    rand_num = np.random.random()
    return rand_sample_helper_slow(x,p,rand_num)
    #return rand_sample_helper(x,p,0,len(x)-1,rand_num)



def random_edge(M, n_walks):
    N = M.shape[0]
    # xv, yv = np.meshgrid(np.arange(N), np.arange(N))
    # xv = xv.reshape(N*N,1)
    # yv = yv.reshape(N*N,1)
    # xv = [list(x) for x in xv]
    # yv = [list(x) for x in yv]
    # coords = zip(yv,xv)
    walks = []
    for i in range(n_walks):
        #pick a column
        col = np.random.choice(range(N), p=sum(M))
        row_dist = M[:,col]/float(sum(M[:,col]))
        row = np.random.choice(range(N), p=row_dist)
        #p = M.flatten()
        #edge_num = rand_sample(range(N*N),p)
        #prob = p[edge_num]
        #edge_num = list(p).index(prob)
        #v = coords[row][0][0]
        #w = coords[col][1][0]
        edge = [row, col]
        walks = walks+edge
    return np.array(walks)


def comp_transition(edges,p,N):
    m = len(p)
    P = np.identity(N)
    sums_a = np.zeros(N)
    sums_b = np.zeros(N)
    for i in range(N):
        edges_i = [[i,j] for j in range(N) if [i,j] in edges]+[[j,i] for j in range(N) if [j,i] in edges]
        indices_i = [x for x in range(m) if edges[x] in edges_i]
        p_i = [p[x] for x in indices_i]
        sums_a[i]=sum(p_i)
    for i in range(len(edges)):
        [u,v] = edges[i]
        P[u][v] = P[u][v] + p[i]
        P[v][u] = P[v][u] + p[i]
        P[u][u] = P[u][u] - p[i]
        P[v][v] = P[v][v] - p[i]
        sums_b[u] = sums_b[u]+p[i]
        sums_b[v] = sums_b[v]+p[i]
    #print('sums')
    #print(sums_a)
    #print(sums_b)
    P[P<.000000000000001]=0
    #print('transition matrix probabilities should add to 1')
    #print(sum(P))
    #print(sum(P.T))
    #print(type(P))
    #print(P-P.T)
    return P

def sub_grad_helper(v,edges,flag):
    g = []
    for [i,j] in edges:
        #if i==14 or j==14:
        #    print('edge')
        #    print(i)
        #    print(j)
        #    print(flag*((v[i]-v[j])**2))
        g.append(flag*((v[i]-v[j])**2))
    return g

def comp_sub_gradient(P,edges):
    U,V = linalg.eig(P)
    idx = U.argsort()[::-1]
    U = U[idx]
    V = V[:,idx]
    if U[1] > -1*U[len(U)-1]:
        g = sub_grad_helper(V[1],edges,-1)
        s = U[1]
    else:
        g = sub_grad_helper(V[len(U)-1],edges,1)
        s = -U[len(U)-1]
    return g, s

def metropolis_hastings(A,edges):
    D = 1./sum(A)
    p = [min([D[i],D[j]]) for [i,j] in edges]
    return p

def vec_tomat(data,coords,N):
    numEntries = len(data)
    P = np.zeros((N,N))
    for x in range(numEntries):
        [i,j] = coords[x]
        P[i][j] = data[x]
        P[j][i] = data[x]
    return P



def fast_mix_transition_matrix_sg(A):
    edges = np.array(A.nonzero()).T
    node_ixs = np.unique(edges[:, 0], return_index=True)[1]
    A = A.toarray()
    edges = [list([x,y]) for [x,y] in edges]
    edges = [[x,y] for [x,y] in edges if x < y]
    N = len(node_ixs)
    #get feasible chain
    p = metropolis_hastings(A,edges)
    P = comp_transition(edges,p,N)
    s_prev = 1000000
    #print('computing fast mix matrix')
    for k in range(1,500):
        #sub-gradient step
        g,s = comp_sub_gradient(P,edges)
        print('mix time')
        print(s)
        #if s-s_prev > .001:
        #    P = P_prev
        #    s=s_prev
        #    break
        alpha_k = 1/np.sqrt(k)
        #norm g
        #print(alpha_k)
        #print(linalg.norm(g))
        g = g/linalg.norm(g)
        delta = -1*alpha_k*g
        #print('change')
        #print(delta[bottleneck_num])
        p = p - alpha_k*g
        #print('new bottleneck')
        #print(p[bottleneck_num])
        M = vec_tomat(p,edges,N)
        #print('total probabilities')
        T = np.sum(M,axis=0)
        #print(T)
        idx = T.argsort()
        #print(idx) 

        v_probs = np.sum(M,axis=0)
        #print(np.sum(M,axis=0))
        #print(np.sum(M,axis=1))

        #print(M[14])
        #print(M[15])
        m = len(p)
        #project on to non-negative
        p = [x if x >0 else 0 for x in p]
        #project so transitions form probability distribution

        for i in idx:
            #if i==14 or i==15:
                ##print('fixing probabilities close to i')
                #print(i)
            #all edges adjacent to i
            edges_i = [[i,j] for j in range(N) if [i,j] in edges]+[[j,i] for j in range(N) if [j,i] in edges]
            neighbors_i = [j for j in range(N) if [i,j] in edges] + [j for j in range(N) if [j,i] in edges]
            #if i==14 or i==15:
                #print(edges_i)
            indices_i = [x for x in range(m) if edges[x] in edges_i]
            indices_i_all = [x for x in range(m) if edges[x] in edges_i]
            #all neighbors of i probabilities
            p_i = [p[x] for x in indices_i]
            p_i_all = [p[x] for x in indices_i]
            totalWeight_ = sum(p_i)
            over = totalWeight_ > 1.0
            while over:
                #print('trying to get p down to 1')
                #only the indices corresponding to edges adjacent to i
                #with positive transition
                #indices_i = [x for x in indices_i if p[x]>0]
                #p_i = [p[x] for x in indices_i]
                #the bigger the ones, the less weight
                #want to divide up the remaining weight using the weights in p_star
                #print(neighbors_i)
                weight = [v_probs[j] for  j in neighbors_i]
                #print(weight)
                weight = [1/j for j in weight]
                #print(weight)
                #zero out the ones that have no edge anymore
                for j in range(len(weight)):
                    if p_i_all[j]==0:
                        weight[j]=0
                #weight = np.multiply(weight,p_i_all)
                Z = sum(weight)
                weight = [w/float(Z) for w in weight]
                #print(p_i_all)
                #print('final weights')
                #print(weight)
                #print(p_i_all)
                #print('new in theory')
                #nt = []
                #for x in indices_i_all:
                #    nt.append(p[x]-(weight[index]*targetRemove))
                #print(nt)

                

                targetRemove = sum(p_i)-1



                #if min(p_i) < (sum(p_i)-1)/len(indices_i):
                #    delta = min(p_i)
                index = 0
                for x in indices_i_all:
                    p[x] = max(0,p[x]-(weight[index]*targetRemove))
                    index = index+1


                #else:
                    


                #delta = min(min(p_i),((sum(p_i)-1)/len(indices_i)))
                #delta = 1/sum(p_i)
                #if i==14 or i==15:
                #    print('delta update')
                #    print(delta)
                #    print('scale factor')
                #    print(delta)
                indices_i = [x for x in indices_i if p[x]>0]
                p_i = [p[x] for x in indices_i]
                p_i_all = [p[x] for x in indices_i_all]
                totalWeight_ = sum(p_i_all)
                #print(totalWeight_)
                over = totalWeight_ > 1.0
        s_prev = s
        P_prev = P
        P = comp_transition(edges,p,N)
        print('bottleneck')
        print(P[4][4])
        tmp = P[4]
        print(np.mean(tmp[tmp>0]))
        tmp = P[5]
        print(np.mean(tmp[tmp>0]))
        #print('average')
        #print(np.average(P))
        if P.min() < 0:
            print(hi)
    return P, [float(s)]
    
def fast_mix_transition_matrix_slem(A):
    W = A.todense()
    G = nx.from_numpy_matrix(W)
    sol = optimize (G , W, max_iter =20)
    mu = sol ['f']
    p = sol ['p']
    i = sol ['iter']
    P = tp_matrix (G , p )
    #print('fmm')
    #for k in range(W.shape[0]):
    #    print(P[:,k].sum())
    return P, mu

def fast_mix_transition_matrix(A):
        edges = np.array(A.nonzero()).T
        node_ixs = np.unique(edges[:, 0], return_index=True)[1]
        A = A.toarray()
        edges = [list([x,y]) for [x,y] in edges]
        N = len(node_ixs)
        #adding the self-loops if these don't exist. if they are fractional,
        #will not do anything 
        for i in range(N):
            if list([i,i]) not in edges:
                edges.append([i,i])
            A[i][i]=1.0
        p = pic.Problem()
        M = cvx.matrix(np.ones([N,N])*1/N)
        I = cvx.matrix(np.identity(N))
        ones = cvx.matrix(np.ones((N,1)))
        M = pic.new_param('M',M)
        I = pic.new_param('I',I)
        ones = pic.new_param('ones',ones)
        P = p.add_variable('P',(N,N),vtype='symmetric')
        s = p.add_variable('s',1)
        p.add_constraint(M - s*I << P)
        p.add_constraint(M + s*I >> P)
        p.add_constraint(P > 0)
        p.add_constraint(P*ones == ones)
        all_pairs = list(itertools.combinations_with_replacement(range(N),2))
        all_pairs = [[x,y] for (x,y) in all_pairs]
        non_edges = [[x,y] for [x,y] in all_pairs if [x,y] not in edges]
        p.add_list_of_constraints([P[i,j]==0 for (i,j) in non_edges], 'i,j', 'non_edges')
        p.add_list_of_constraints([P[int(l),int(z)] < A[int(l)][int(z)] for (l,z) in edges], 'l,z', 'edges')
        p.set_objective('min',s)
        print(p)
        print('SOLVING')
        p.solve(verbose=1)
        P = np.array(P.value)
        print(s.value)
        print(type(s.value))
        s = np.array(s.value)
        print(s[0])
        P[P<.000000000000001]=0
        return P, s[0]

def graph_entropy_matrix(A):
    E = np.zeros(A.shape)
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            if (A[i,j]<.0000000000001) or (A[i,j]>.9999999999):
                E[i,j]=0
            else:
                E[i,j] = spst.entropy([A[i,j],1-A[i,j]])
    return E


class RandomWalker:
    """
    Helper class to generate random walks on the input adjacency matrix.
    """
    def __init__(self, adj, rw_len, p=1, q=1, batch_size=128):
        self.adj = adj
        #if not "lil" in str(type(adj)):
        #    warnings.warn("Input adjacency matrix not in lil format. Converting it to lil.")
        #    self.adj = self.adj.tolil()

        self.rw_len = rw_len
        self.p = p
        self.q = q
        A = adj.todense()
        self.s = []
        self.edges = np.array(self.adj.nonzero()).T
        self.node_ixs = np.unique(self.edges[:, 0], return_index=True)[1]
        N = len(self.node_ixs)
        self.RW = randWalkMatrix(self.edges, self.node_ixs)
        s_test = solveStationary(np.transpose(self.RW))
        self.s = [x.tolist()[0][0] for x in s_test]
        

        self.batch_size = batch_size
        self.P, self.fmm_time = fast_mix_transition_matrix_slem(self.adj)
        
        #self.M = genEdgeDist(self.edges, self.node_ixs, self.rw_len, self.p, self.q)
        
        
        
        test = []
        

        self.Combo = .5*self.P + .5*self.RW 
        combo_stationary =solveStationary(np.transpose(self.Combo))

        self.combo_stationary = [x.tolist()[0][0] for x in combo_stationary]
        self.uniform = np.ones(N)*(1/float(N))
        self.RW_ex = expected_walk_w_matrix(self.node_ixs, self.rw_len, self.RW, self.uniform)
        #self.fmm_ex = expected_walk_w_matrix(self.node_ixs,self.rw_len, self.P, uniform)
        self.RW_ex_correct = expected_walk_w_matrix(self.node_ixs, self.rw_len, self.RW, self.s)
        #self.M_fmm = self.P/(float(len(self.node_ixs)))
        


    def walk(self):
        while True:
            #yield random_walk(self.edges, self.node_ixs, self.rw_len, self.p, self.q, self.batch_size).reshape([-1, self.rw_len])
            yield random_walk_w_matrix(self.edges, self.node_ixs, self.rw_len, self.RW, self.batch_size, self.s).reshape([-1, self.rw_len])

    def fast_mix_walk(self):
        while True:
            yield random_walk_w_matrix(self.edges, self.node_ixs, self.rw_len, self.P, self.batch_size, self.uniform).reshape([-1, self.rw_len])

    def combo_walk(self):
        while True:
            yield random_walk_w_matrix(self.edges, self.node_ixs, self.rw_len, self.Combo, self.batch_size, self.combo_stationary).reshape([-1, self.rw_len])

    def edgeProb(self):
        return genEdgeDist(self.edges, self.node_ixs, self.rw_len, self.p, self.q)

    def edgeProb_fmm(self):
        return self.M_fmm
    
    def edge(self):
        while True:
            yield random_walk_w_matrix_edges(self.edges, self.node_ixs, self.rw_len, self.RW, self.batch_size, self.s).reshape([-1, 2])

    def edge_fmm(self):
        while True:
            yield random_walk_w_matrix_edges(self.edges, self.node_ixs, self.rw_len, self.P, self.batch_size, self.uniform).reshape([-1, 2])

    



def edge_overlap(A, B):
    """
    Compute edge overlap between input graphs A and B, i.e. how many edges in A are also present in graph B. Assumes
    that both graphs contain the same number of edges.

    Parameters
    ----------
    A: sparse matrix or np.array of shape (N,N).
       First input adjacency matrix.
    B: sparse matrix or np.array of shape (N,N).
       Second input adjacency matrix.

    Returns
    -------
    float, the edge overlap.
    """

    return ((A == B) & (A == 1)).sum()

def graph_from_transitions(T, edges, n_edges, N):
    mat = np.zeros([N,N])
    rand_edge = np.random.choice(range(len(edges)))
    (i,j) = edges[rand_edge]
    mat[i,j] = 1
    mat[j,i] = 1
    edges_placed = 1
    while edges_placed < n_edges:
        k = np.random.choice(range(N),p=T[i,j,:])
        i = j
        j = k
        if mat[i,j] == 0:
            mat[i,j] = 1
            mat[j,i] = 1
            edges_placed = edges_placed + 1
    return mat

def solveStationary( A ):
    """ x = xA where x is the answer
    x - xA = 0
    x( I - A ) = 0 and sum(x) = 1
    """
    n = A.shape[0]
    a = np.eye( n ) - A
    a = np.vstack( (a.T, np.ones( n )) )
    b = np.matrix( [0] * n + [ 1 ] ).T
    return np.linalg.lstsq( a, b )[0]




def graph_from_scores(scores, n_edges):
    """
    Assemble a symmetric binary graph from the input score matrix. Ensures that there will be no singleton nodes.
    See the paper for details.

    Parameters
    ----------
    scores: np.array of shape (N,N)
            The input transition scores.
    n_edges: int
             The desired number of edges in the target graph.

    Returns
    -------
    target_g: symmettic binary sparse matrix of shape (N,N)
              The assembled graph.

    """

    if  len(scores.nonzero()[0]) < n_edges:
        print('modes collapsing')
        return symmetric(scores) > 0

    target_g = np.zeros(scores.shape) # initialize target graph
    scores_int = scores.toarray().copy() # internal copy of the scores matrix
    scores_int[np.diag_indices_from(scores_int)] = 0  # set diagonal to zero
    degrees_int = scores_int.sum(0)   # The row sum over the scores.

    N = scores.shape[0]

    for n in np.random.choice(N, replace=False, size=N): # Iterate the nodes in random order

        row = scores_int[n,:].copy()
        if row.sum() == 0:
            continue

        probs = row / row.sum()

        target = np.random.choice(N, p=probs)
        target_g[n, target] = 1
        target_g[target, n] = 1


    diff = np.round((n_edges - target_g.sum())/2)
    if diff > 0:

        triu = np.triu(scores_int)
        triu[target_g > 0] = 0
        triu[np.diag_indices_from(scores_int)] = 0
        triu = triu / triu.sum()

        triu_ixs = np.triu_indices_from(scores_int)
        extra_edges = np.random.choice(triu_ixs[0].shape[0], replace=False, p=triu[triu_ixs], size=int(diff))
        target_g[(triu_ixs[0][extra_edges], triu_ixs[1][extra_edges])] = 1
        target_g[(triu_ixs[1][extra_edges], triu_ixs[0][extra_edges])] = 1

    target_g = symmetric(target_g)
    return target_g


def symmetric(directed_adjacency, clip_to_one=True):
    """
    Symmetrize the input adjacency matrix.
    Parameters
    ----------
    directed_adjacency: sparse matrix or np.array of shape (N,N)
                        Input adjacency matrix.
    clip_to_one: bool, default: True
                 Whether the output should be binarized (i.e. clipped to 1)

    Returns
    -------
    A_symmetric: sparse matrix or np.array of the same shape as the input
                 Symmetrized adjacency matrix.

    """

    A_symmetric = directed_adjacency + directed_adjacency.T
    if clip_to_one:
        A_symmetric[A_symmetric > 1] = 1
    return A_symmetric

def squares(g):
    """
    Count the number of squares for each node
    Parameters
    ----------
    g: igraph Graph object
       The input graph.

    Returns
    -------
    List with N entries (N is number of nodes) that give the number of squares a node is part of.
    """

    cliques = g.cliques(min=4, max=4)
    result = [0] * g.vcount()
    for i, j, k, l in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
        result[l] += 1
    return result


def statistics_degrees(A_in):
    """
    Compute min, max, mean degree

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_numpy_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def statistics_square_count(A_in):
    """
    Compute the square count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Square count
    """

    A_igraph = igraph.Graph.Adjacency((A_in > 0).tolist()).as_undirected()
    return int(np.sum(squares(A_igraph)) / 4)


def statistics_power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """

    degrees = A_in.sum(axis=0)
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees),1)).power_law.alpha


def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    degrees = A_in.sum(axis=0)
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
                                                                                                               n + 1) / n
    return float(G)


def statistics_edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Rel. edge distribution entropy
    """

    degrees = A_in.sum(axis=0)
    m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees+.0001) / (2 * float(m))))
    return H_er





def sp(A):
    G = nx.from_numpy_matrix(A).to_undirected()
    length_dict = dict(nx.all_pairs_shortest_path_length(G,cutoff=100))
    shortestPathLengths = []
    vertices = length_dict.keys()
    for j in vertices:
        shortestPathLengths = shortestPathLengths + length_dict[j].values()
    return sorted(shortestPathLengths)

def cc(A):
    G = nx.from_numpy_matrix(A).to_undirected()
    clustering_coefficient = nx.clustering(G)
    clustering_coefficient = sorted(clustering_coefficient.values())
    return sorted(clustering_coefficient)

def genDiscDist_simple(values,slist):
    numValues = len(values)
    lenList = len(slist)
    result = [0]*numValues
    for v in range(numValues):
        cnt = slist.count(values[v])
        result[v]=cnt/float(lenList) 
    return result

def emdLinear(hist1,hist2,values):
    numBins = len(hist1)
    dist = 0
    totalDist = 0
    for i in range(numBins-1):
        dist = (hist1[i]+dist)-hist2[i]
        totalDist = totalDist+(abs(dist)*(values[i+1]-values[i]))
    return totalDist

def emd(x,y):
    totalValues = sorted(list(set().union(x,y)))
    x_dist = genDiscDist_simple(totalValues,sorted(x))
    y_dist = genDiscDist_simple(totalValues,sorted(y))
    return emdLinear(x_dist,y_dist,totalValues)

def specGap(A):
    G = nx.from_numpy_matrix(A).to_undirected()
    L = nx.normalized_laplacian_matrix(G).todense()
    w,v = np.linalg.eig(L)
    w = sorted(w)
    return w[1]

def spectrum(A):
    G = nx.from_numpy_matrix(A).to_undirected()
    L = nx.normalized_laplacian_matrix(G).todense()
    w,v = np.linalg.eig(L)
    w = sorted(w)
    return w


def degree_sequence(A):
    G = nx.from_numpy_matrix(A).to_undirected()
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    return degree_sequence





def compute_graph_statistics(A_in):
    """

    Parameters
    ----------
    A_in: sparse matrix
          The input adjacency matrix.
    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Size of the largest connected component (LCC)
             * Wedge count
             * Claw count
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Assortativity
             * Clustering coefficient
             * Number of connected components
    """

    A = A_in.copy()

    assert ((A == A.T).all())
    A_graph = nx.from_numpy_matrix(A).to_undirected()

    statistics = {}

    d_max, d_min, d_mean = statistics_degrees(A)

    # Degree statistics
    statistics['d_max'] = d_max
    statistics['d_min'] = d_min
    statistics['d'] = d_mean

    # largest connected component
    LCC = statistics_LCC(A)

    statistics['LCC'] = LCC.shape[0]
    # wedge count
    statistics['wedge_count'] = statistics_wedge_count(A)

    # claw count
    statistics['claw_count'] = statistics_claw_count(A)

    # triangle count
    statistics['triangle_count'] = statistics_triangle_count(A)

    # Square count
    statistics['square_count'] = statistics_square_count(A)

    # power law exponent
    statistics['power_law_exp'] = statistics_power_law_alpha(A)

    # gini coefficient
    statistics['gini'] = statistics_gini(A)

    # Relative edge distribution entropy
    statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)

    # Assortativity
    statistics['assortativity'] = nx.degree_assortativity_coefficient(A_graph)

    # Clustering coefficient
    statistics['clustering_coefficient'] = 3 * statistics['triangle_count'] / statistics['claw_count']

    # Number of connected components
    statistics['n_components'] = connected_components(A)[0]

    #statistics['sp_emd'] = sp_emd(A)

    return statistics


