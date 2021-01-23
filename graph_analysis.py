import numpy as np
import networkx as nx
from utils import load_data
import matplotlib.pyplot as plt
from matplotlib import pylab
import heapq
from networkx.algorithms import community
import itertools
from mpl_toolkits.basemap import Basemap as Basemap
from collections import Counter

def save_graph(graph, file_name, node_sizes = None, node_colors = None,
               cmap = plt.cm.Reds, idx = None, 
               idx_to_latitude_map = None, idx_to_longitude_map = None):
    
    m = Basemap(
            projection = 'merc',
            llcrnrlon = -15, # west
            llcrnrlat = 30, # south
            urcrnrlon = 90, # east
            urcrnrlat = 73, # north
            lat_ts = 0,
            resolution = 'l',
            suppress_ticks = True)
    #initialze Figure
    plt.figure(num = None, figsize = (20, 20), dpi = 80)
    plt.axis('off')
    fig = plt.figure(1)
    
    if idx_to_latitude_map is None or idx_to_longitude_map is None or idx is None:
        pos = nx.random_layout(graph, seed = 42)
    else:
        lats = [idx_to_latitude_map[i] for i in idx]
        lons = [idx_to_longitude_map[i] for i in idx]
        mx, my = m(lons,lats)
        pos = {i: (mx[i - 1], my[i - 1]) for i in idx}
        
    nx.draw_networkx_nodes(graph, pos = pos, edgecolors = 'black', cmap = cmap, node_size = node_sizes, node_color = node_colors)
    nx.draw_networkx_edges(graph, pos = pos, edge_color = 'yellow')
    
    if not (idx_to_latitude_map is None or idx_to_longitude_map is None or idx is None):
        m.drawcountries()
        m.bluemarble()

    plt.savefig(file_name, bbox_inches = "tight")
    pylab.close()
    del fig

def main():
    idx, cities, idx_to_city_map, city_to_idx_map, edges, adj, \
    idx_to_latitude_map, idx_to_longitude_map = load_data()
    
    edges_tuples = [(edges[i][0], edges[i][1]) for i in range(edges.shape[0])]
    
    g = nx.Graph()
    g.add_nodes_from(idx_to_city_map)
    g.add_edges_from(edges_tuples)
    
    print('Number of nodes: ', g.number_of_nodes())
    print('Number of edges: ', g.number_of_edges())
    print('')
    
    S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
    giant = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    
    print('Number of connected components: ', len(S))
    
    node_degrees = g.degree()
    
    avg_degree = np.mean([deg[1] for deg in node_degrees])
    max_degree = np.max([deg[1] for deg in node_degrees])
    
    diameter = max(nx.diameter(s) for s in S)
    
    density = nx.density(g)
    
    avg_clustering_coef = nx.average_clustering(g)
    avg_path_len = np.mean([nx.average_shortest_path_length(s) for s in S])
    
    print('Average node degree: ', avg_degree)
    print('Max node degree: ', max_degree)
    print('Graph diameter: ', diameter)
    print('Graph density: ', density)
    print('Clustering coefficient: ', avg_clustering_coef)
    print('Average path length: ', avg_path_len)
    print('')
    
    degs = [val for (node, val) in g.degree()]
    degrees, count = zip(*Counter(degs).items())
    plt.scatter(degrees, count)
    plt.title("Degree distribution")
    plt.xlabel("Degree")
    plt.ylabel("Value")
    plt.show()
    
    '''
    degree_centrality = nx.degree_centrality(g)
    betweenness_centrality = nx.betweenness_centrality(g)
    closeness_centrality = nx.closeness_centrality(g)
    eigenvector_centrality = nx.eigenvector_centrality(g, max_iter = 1000)
    
    degree_centrality_10 = heapq.nlargest(10, degree_centrality, key=degree_centrality.get)
    betweenness_centrality_10 = heapq.nlargest(10, betweenness_centrality, key=betweenness_centrality.get)
    closeness_centrality_10 = heapq.nlargest(10, closeness_centrality, key=closeness_centrality.get)
    eigenvector_centrality_10 = heapq.nlargest(10, eigenvector_centrality, key=eigenvector_centrality.get)
    
    degree_centrality_10_city = {idx_to_city_map[i]: degree_centrality.get(i) for i in degree_centrality_10}
    betweenness_centrality_10_city = {idx_to_city_map[i]: betweenness_centrality.get(i) for i in betweenness_centrality_10}
    closeness_centrality_10_city = {idx_to_city_map[i]: closeness_centrality.get(i) for i in closeness_centrality_10}
    eigenvector_centrality_10_city = {idx_to_city_map[i]: eigenvector_centrality.get(i) for i in eigenvector_centrality_10}
    
    print('Degree centrality: ', degree_centrality_10_city)
    print('')
    print('Betweenness centrality: ', betweenness_centrality_10_city)
    print('')
    print('Closeness centrality: ', closeness_centrality_10_city)
    print('')
    print('Eigenvector centrality: ', eigenvector_centrality_10_city)
    print('')
    
    
    node_sizes = [20 + degree_centrality[i] * 20000 for i in idx]
    node_colors = [70 + degree_centrality[i] * 100 for i in idx] 
    save_graph(g, "graph_degree_centrality.pdf", node_sizes, node_colors, idx = idx, 
               idx_to_latitude_map = idx_to_latitude_map, idx_to_longitude_map = idx_to_longitude_map)
    
    node_sizes = [20 + betweenness_centrality[i] * 1000 for i in idx]
    node_colors = [70 + betweenness_centrality[i] * 100 for i in idx] 
    save_graph(g, "graph_betweenness_centrality.pdf", node_sizes, node_colors, idx = idx, 
               idx_to_latitude_map = idx_to_latitude_map, idx_to_longitude_map = idx_to_longitude_map)
    
    node_sizes = [10 + closeness_centrality[i] * 1000 for i in idx]
    node_colors = [50 + closeness_centrality[i] * 50 for i in idx] 
    save_graph(g, "graph_closeness_centrality.pdf", node_sizes, node_colors, idx = idx, 
               idx_to_latitude_map = idx_to_latitude_map, idx_to_longitude_map = idx_to_longitude_map)
    
    node_sizes = [20 + eigenvector_centrality[i] * 1000 for i in idx]
    node_colors = [70 + eigenvector_centrality[i] * 100 for i in idx] 
    save_graph(g, "graph_eigenvector_centrality.pdf", node_sizes, node_colors, idx = idx, 
               idx_to_latitude_map = idx_to_latitude_map, idx_to_longitude_map = idx_to_longitude_map)
    
    k = 6
    colors = ['red', 'blue', 'green', 'purple', 'cyan', 'orange']
    
    comp = community.girvan_newman(giant)
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    largest_communities = list(limited)[-1]
    
    node_colors = []
    for i in idx:
        added = False
        for k in range(0, len(largest_communities)):
            if i in largest_communities[k]:
                node_colors.append(colors[k])
                added = True
                
        if not added:
            node_colors.append('gray')
    save_graph(g, "graph_communities_6_girvan_newman.pdf", node_colors = node_colors, idx = idx, 
               idx_to_latitude_map = idx_to_latitude_map, idx_to_longitude_map = idx_to_longitude_map)
    
    for c in largest_communities:
        print("Community: ")
        cities_in_c = [idx_to_city_map[city_idx] for city_idx in c]
        print(cities_in_c)
        print('')
    '''
    
    '''communities_generator = community.label_propagnation(g)
    c = list(communities_generator)
    print(len(c))
    
    c = community.greedy_modularity_communities(g)
    print(len(c))
    c.sort(key = len)
    largest_communities = c[-8:]''' 
    
if __name__ == '__main__':
    main()