from matplotlib.colors import same_color
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from fairmotion.ops import conversions

class Interaction(object):
    def __init__(self,vert,self_cnt=0) -> None:
        self.vert = vert
        self.self_cnt = self_cnt
    def build_interaction_graph(self, type="interaction_mesh"):
        edges = []
        scales = []
        if type == "interaction_mesh":
            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    edge = [i,neighbor_vert]
                    edges.append(edge)
            edges = np.array(edges).T
        elif type == "fully_connected":
            full_matrix = np.ones((self.vert.shape[0],self.vert.shape[0]))
            matrix = coo_matrix(full_matrix)
            edges = np.array([matrix.row,matrix.col])
        elif type == "full_bipartite":
            full_matrix = np.ones((self.vert.shape[0],self.vert.shape[0]))
            full_matrix[:self.self_cnt,:self.self_cnt] = 0
            full_matrix[self.self_cnt:,self.self_cnt:] = 0
            matrix = coo_matrix(full_matrix)
            edges = np.array([matrix.row,matrix.col])     
        elif type == "interaction_mesh_bipartite":
            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    edge = [i,neighbor_vert]
                    edges.append(edge)
            edges = np.array(edges).T   
            cross_edges_idx = np.where((((edges[0]>self.self_cnt) & (edges[1]<self.self_cnt)) | ((edges[0]<self.self_cnt) & (edges[1]>self.self_cnt))))
            
            edges = edges[:,cross_edges_idx[0]]
        elif type == 'interaction_mesh_filtered':
            filted_pairs = [

                (0,8), # hip - right upper leg
                (0,11), # hip - left upper leg
                (8,0),  # hip - right upper leg
                (11,0), # hip - left upper leg
                (8,11), # left upper leg - right upper leg
                (11,8), # left upper leg - right upper leg
                
                (15,23), # hip - right upper leg (oppo)
                (15,26), # hip - left upper leg (oppo)
                (23,15), # hip - right upper leg (oppo)
                (26,15), # hip - left upper leg (oppo)
                (23,26), # left upper leg - right upper leg (oppo)
                (26,23), # left upper leg - right upper leg (oppo)

            ]
            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    if (i,neighbor_vert) in filted_pairs:
                        continue
                    edge = [i,neighbor_vert]
                    edges.append(edge)
            edges = np.array(edges).T   
        elif type == 'interaction_mesh_filtered_2':
            filted_pairs = [

                (0,8), # hip - right upper leg
                (0,11), # hip - left upper leg
                (8,0),  # hip - right upper leg
                (11,0), # hip - left upper leg
                (8,11), # left upper leg - right upper leg
                (11,8), # left upper leg - right upper leg
                (14,2), # head - right shoulder
                (14,5), # head - left shoulder
                (2,14), # right shoulder - head
                (5,14), # left shoulder - head

                (15,23), # hip - right upper leg (oppo)
                (15,26), # hip - left upper leg (oppo)
                (23,15), # hip - right upper leg (oppo)
                (26,15), # hip - left upper leg (oppo)
                (23,26), # left upper leg - right upper leg (oppo)
                (26,23), # left upper leg - right upper leg (oppo)

                (29,17), # head - right shoulder (oppo)
                (29,20), # head - left shoulder (oppo)
                (17,29), # right shoulder - head (oppo)
                (20,29), # left shoulder - head (oppo)

            ]
            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    if (i,neighbor_vert) in filted_pairs:
                        continue
                    edge = [i,neighbor_vert]
                    edges.append(edge)
            edges = np.array(edges).T    
        elif type == "interaction_mesh_filtered_3":
            filted_pairs = [

                (0,8), # hip - right upper leg
                (0,11), # hip - left upper leg
                (8,0),  # hip - right upper leg
                (11,0), # hip - left upper leg
                (8,11), # left upper leg - right upper leg
                (11,8), # left upper leg - right upper leg
                (14,2), # head - right shoulder
                (14,5), # head - left shoulder
                (2,14), # right shoulder - head
                (5,14), # left shoulder - head

                (15,23), # hip - right upper leg (oppo)
                (15,26), # hip - left upper leg (oppo)
                (23,15), # hip - right upper leg (oppo)
                (26,15), # hip - left upper leg (oppo)
                (23,26), # left upper leg - right upper leg (oppo)
                (26,23), # left upper leg - right upper leg (oppo)

                (29,17), # head - right shoulder (oppo)
                (29,20), # head - left shoulder (oppo)
                (17,29), # right shoulder - head (oppo)
                (20,29), # left shoulder - head (oppo)
            ]   

            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    if (i,neighbor_vert) in filted_pairs:
                        continue
                    edge = [i,neighbor_vert]
                    edges.append(edge)
                    
            edges = np.array(edges).T   
            cross_edges_idx = np.where(((edges[0]<self.self_cnt) & (edges[1]<self.self_cnt)))
            
            edges = edges[:,cross_edges_idx[0]]
        elif type == "interaction_mesh_filtered_4":
            filted_pairs = [
            
                (0,8), # hip - right upper leg
                (0,11), # hip - left upper leg
                (8,0),  # hip - right upper leg
                (11,0), # hip - left upper leg
                (8,11), # left upper leg - right upper leg
                (11,8), # left upper leg - right upper leg
                (14,2), # head - right shoulder
                (14,5), # head - left shoulder
                (2,14), # right shoulder - head
                (5,14), # left shoulder - head
                ## Bone Edges
                # (2,3),
                # (3,4),
                # (5,6),
                # (6,7),
                # (8,9),
                # (9,10),
                # (11,12),
                # (12,13),

                # (3,2),
                # (4,3),
                # (6,5),
                # (7,6),
                # (9,8),
                # (10,9),
                # (12,11),
                # (13,12),
            ]   

            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    if (i,neighbor_vert) in filted_pairs:
                        continue
                    edge = [i,neighbor_vert]
                    edges.append(edge)
                    
            edges = np.array(edges).T   
            cross_edges_idx = np.where(((edges[0]<self.self_cnt) & (edges[1]<self.self_cnt)))
            
            # edges = edges[:,cross_edges_idx[0]]
            edges = [
                [0],
                [13]
                ]
        elif type == "interaction_mesh_remove_self":
            filtered_verts = [15,16,17,18]
            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    if i in filtered_verts and neighbor_vert in filtered_verts:
                        continue
                    edge = [i,neighbor_vert]
                    edges.append(edge)
            edges = np.array(edges).T
        elif type == "interaction_mesh_remove_self_obj_vert":
            filtered_verts = [15,16,17,18,19,20,21,22]
            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    if i in filtered_verts and neighbor_vert in filtered_verts:
                        continue
                    edge = [i,neighbor_vert]
                    edges.append(edge)
            edges = np.array(edges).T     
        return edges
