import numpy as np
import scipy.sparse as sp

def load_data(path = 'data/', dataset = 'euroroad'):
    print('Loading dataset')
    
    # get vertices, split to idx and cities
    idx_cities_lat_long = np.genfromtxt("{}{}.vertices".format(path, dataset), dtype = np.dtype(str), delimiter = '"')
    idx = np.array(idx_cities_lat_long[:, 0], dtype = np.int32)
    cities = np.array(idx_cities_lat_long[:, 1], dtype = np.dtype(str))
    latitudes = np.array(idx_cities_lat_long[:, 2], dtype = np.float32)
    longitudes = np.array(idx_cities_lat_long[:, 3], dtype = np.float32)
    
    idx_to_city_map = {(i + 1): cities[i] for i in range(len(idx))} 
    city_to_idx_map = {cities[i]: (i + 1) for i in range(len(idx))}
    idx_to_latitude_map = {(i + 1): latitudes[i] for i in range(len(idx))}
    idx_to_longitude_map = {(i + 1): longitudes[i] for i in range(len(idx))}
    
    print(idx_cities_lat_long)
    print('')

    # get edges, build adj matrix
    edges_unordered = np.genfromtxt("{}{}.edges".format(path, dataset), dtype = np.int32)
    edges = np.array(list(obj for obj in edges_unordered.flatten()), dtype = np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), dtype=np.float32)        
    
    # build symmetric adjacency matrix, multiply point-wise
    # adj.T > adj gets matrix with elements that adj.T has and adj doesn't
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    
    return idx, cities, idx_to_city_map, city_to_idx_map, edges, adj, idx_to_latitude_map, idx_to_longitude_map
    

def normalize_sparse(mx):
    """ Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

if __name__ == '__main__':
    load_data()
    