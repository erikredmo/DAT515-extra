# Cluster extra lab
import csv
import networkx as nx
import matplotlib.pyplot as plt
from haversine import haversine
import random

AIRPORT_FILE = '/Users/erikredmo/Code/DAT515/clustering/csvfiles/airports.dat.csv'
ROUTES_FILE = '/Users/erikredmo/Code/DAT515/clustering/csvfiles/routes.dat.csv'

###### Lab functions ######
def mk_airportdict(FILE):
    with open(FILE, newline='') as csvfile:
        reader = csv.reader(csvfile)
        airport_dict = {}
        pos_dict = {}
        for row in reader:
            try:
                airport_dict[row[0]] = {'Name': row[1],
                                        'City': row[2],
                                        'Country': row[3],
                                        'IATA': row[4],
                                        'ICAO': row[5],
#                                        'Latitude': row[6],
#                                        'Longitude': row[7],
                                        'Position': (float(row[6]), float(row[7])), # It's always nice to have position as x, y. (lat, lon) for haversine
                                        'Altitude': row[8],
                                        'Timezone': row[9],
                                        'DST': row[10],
                                        'Tz database time zone': row[11],
                                        'Type': row[12],
                                        'Source': row[13]}
                if len(row[4]) == 3:
                    iata = row[4]
                    row_id = row[0]
                    airport_dict[iata] = row_id # To change time complexity from quadratic to linear? Here we can just call for airport_dict[IATA] when finding our ID
                pos_dict[row[0]] = (float(row[7]), float(row[6])) # It's nice to have positions easily at hand for visualization, (lon, lat) for matlpotlib
            except:
                continue

    return airport_dict, pos_dict

def mk_routeset(FILE, airport_dict):
    with open(FILE, newline='') as csvfile:
        reader = csv.reader(csvfile)
        route_set = set()
        for row in reader:
            try:
                IATA_dep = row[2]
                IATA_dest = row[4]
                dep_id = airport_dict[IATA_dep]
                dest_id = airport_dict[IATA_dest]
                route_set.add((dep_id, dest_id))
#                if IATA_dep != IATA_dest and airport_dict[dep_id]['City'] != airport_dict[dest_id]['City']: # Deleting routes to same city due to readability
#                    route_set.add((dep_id, dest_id))
#                else:
#                    continue
            except:
                continue
                
    return route_set

def mk_routegraph(routeset, airport_dict):
    G = nx.Graph()
    color_list = ['r', 'b', 'g', 'y', 'c', 'm']
    color_to_graph = []
    for edge in routeset:
        try:
            G.add_edge(edge[0], edge[1], weight=compute_geo_distance(edge, airport_dict)) 
            color_to_graph.append(random.choice(color_list))
        except:
            G.remove_edge(edge[0], edge[1]) # Just in case of any awkward errors

    return G, color_to_graph

def k_spanning_tree(G, k):
    mst = nx.algorithms.tree.minimum_spanning_edges(G, algorithm='prim')
    edgelist = list(mst)
    edgelist.sort(key=lambda x:x[2]['weight'], reverse=True)
    edgelist_cut_k_heaviest = edgelist[k:]
    mst_G = mk_routegraph(edgelist_cut_k_heaviest, airports[0])
    
    return mst_G

from sklearn.cluster import KMeans
import numpy as np
def k_means(data, k):
    X = np.array(np.array(edge_pos for pos in data)) #data = pos_dict
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(list(data.values()))
    pos_and_label = []
    for pos_id, label in zip(data.values(), labels):
        pos_and_label.append([pos_id[0], pos_id[1], label])

    return np.array(pos_and_label)










###### My own utility functions ######
def label_color(label): # For colors to kmeans
    colors = []
    for label_number in label:
        if label_number == 0:
            colors.append('tab:blue')
        elif label_number == 1:
            colors.append('tab:orange')
        elif label_number == 2:
            colors.append('tab:green')
        elif label_number == 3:
            colors.append('tab:red')
        elif label_number == 4:
            colors.append('tab:purple')
        elif label_number == 5:
            colors.append('tab:brown')
        elif label_number == 6:
            colors.append('tab:pink')
        elif label_number == 7:
            colors.append('tab:olive')
        elif label_number == 8:
            colors.append('tab:cyan')
        else:
            colors.append('tab:gray')
    
    return np.array(colors)

def compute_geo_distance(id_edge, airport_dict):
    try:
        airport_dep_pos = airport_dict[id_edge[0]]['Position']
        airport_dest_pos = airport_dict[id_edge[1]]['Position']
        return haversine(airport_dep_pos, airport_dest_pos)
    except:
        pass












###### VISUALIZE ######
def visualizenodes():
    pos = airports[1]
    plt.figure(figsize = (20, 7.5))
    pos_list = pos.values()
    lat = []
    lon = []
    for positions in pos_list:
        lat.append(positions[0])
        lon.append(positions[1])

    plt.scatter(lat, lon, s=0.4, alpha=0.8)
    nx.draw_networkx_nodes(graph_routes[0], pos=pos, node_shape = '.' , node_size = 4, alpha = 0.8) 
    plt.show()

def visualizeedges():
    pos = airports[1]
    plt.figure(figsize = (20, 7.5))
    pos_list = pos.values()
    lat = []
    lon = []
    for positions in pos_list:
        lat.append(positions[0])
        lon.append(positions[1])

    plt.scatter(lat, lon, s=0.4, alpha=0.8)
    nx.draw_networkx_edges(graph_routes[0], pos=pos, edge_color=graph_routes[1], width = 0.3, alpha = 0.5) 
    plt.show()

def visualize_k_spanning_tree():
    pos = airports[1]
    plt.figure(figsize = (20, 7.5))
    pos_list = pos.values()
    lat = []
    lon = []
    for positions in pos_list:
        lat.append(positions[0])
        lon.append(positions[1])

    plt.scatter(lat, lon, s=0.4, alpha=0.8)
    nx.draw_networkx_edges(mst_G[0], pos=pos, edge_color=graph_routes[1], width = 1.5, alpha = 0.8)
    plt.show()

def visualize_k_means():
    plt.figure(figsize = (20, 7.5))
    plt.scatter(pos_and_labels[:,0], pos_and_labels[:,1], s=0.4, alpha=0.8, color=label_color(pos_and_labels[:,2]))
    plt.show()












###### MAIN ######
import sys

if __name__ == '__main__':
    if sys.argv[1:] == ['airports']:
        airports = mk_airportdict(AIRPORT_FILE)
        route_set = mk_routeset(ROUTES_FILE, airports[0])
        graph_routes = mk_routegraph(route_set, airports[0])
        visualizenodes()

    elif sys.argv[1:] == ['routes']:
        airports = mk_airportdict(AIRPORT_FILE)
        route_set = mk_routeset(ROUTES_FILE, airports[0])
        graph_routes = mk_routegraph(route_set, airports[0])
        visualizeedges()
    
    elif sys.argv[1:-1] == ['span']:
        k = int(sys.argv[-1])
        airports = mk_airportdict(AIRPORT_FILE)
        route_set = mk_routeset(ROUTES_FILE, airports[0])
        graph_routes = mk_routegraph(route_set, airports[0])
        mst_G = k_spanning_tree(graph_routes[0], k)
        visualize_k_spanning_tree()
    
    elif sys.argv[1:-1] == ['means']:
        k = int(sys.argv[-1])
        if k > 10:
            print('Can only show clusters using 10 different colors.')
        airports = mk_airportdict(AIRPORT_FILE)
        route_set = mk_routeset(ROUTES_FILE, airports[0])
        graph_routes = mk_routegraph(route_set, airports[0])
        pos_and_labels = k_means(airports[1], k)
        visualize_k_means()
    
    else:
        print('Invalid input, please try again')