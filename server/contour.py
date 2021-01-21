'''
The code for generating the contour of current selection

kernel_density: performs 2d histogram-based approximated kde (HAKDE)
marching_squares: perform marching squares algorithm for final contour computation

'''

import numpy as np 
import math

from numba import cuda


def generate(current_selection, current_emb, max_emb, min_emb):

    g_size = 20
    grid_matrix = np.zeros((g_size, g_size))
    grid_sign = np.zeros((g_size, g_size))
    grid_square = np.zeros((g_size - 1, g_size - 1))

    ## smallest block dimension : smaller to bigger, 
    contour_configuration = [
        [],                                      ## 0
        [[[2, 0], [1, 0]]],                      ## 1 
        [[[0, 1], [3, 1]]],                      ## 2
        [[[2, 0], [3, 1]]],                      ## 3
        [[[3, 2], [0, 2]]],                      ## 4
        [[[3, 2], [1, 0]]],                      ## 5
        [[[0, 1], [0, 2]], [[2, 3], [3, 1]]],    ## 6
        [[[3, 2], [3, 1]]],                      ## 7
        [[[1, 3], [2, 3]]],                      ## 8
        [[[3, 1], [1, 0]], [[2, 0], [2, 3]]],    ## 9
        [[[0, 1], [2, 3]]],                      ## 10
        [[[2, 0], [2, 3]]],                      ## 11
        [[[1, 3], [0, 2]]],                      ## 12
        [[[1, 3], [1, 0]]],                      ## 13
        [[[0, 1], [0, 2]]],                      ## 14   
        [],                                      ## 15
    ]
    
    isovalue = kernel_density(current_selection, current_emb, grid_matrix, max_emb, min_emb)
    marching_squares(grid_matrix, grid_sign, grid_square, contour_configuration, isovalue)


## current_selection: current selected points which needs to be density-estimated
## gird_matrix: stores grid value based on kernel density estimation (should be np array, zero-initialized)
## max_emb: max value of the embedding ([x, y])
## min_emb: min value of the embedding ([x, y])
def kernel_density(current_selection, current_emb, grid_matrix, max_emb, min_emb):
    grid_size = grid_matrix.shape[0]

    range_emb = max_emb - min_emb

    def x_scale(val):
        scaled = (val - max_emb[0]) / range_emb[0]
        scaled = round((grid_size - 1) * scaled)
        return scaled

    def y_scale(val):
        scaled = (val - max_emb[1]) / range_emb[1]
        scaled = round((grid_size - 1) * scaled)
        return scaled

    ## ALL steps can be parallelized

    ## generate grid-wise (historgram-based) distribution of the points
    grid_distribution = np.zeros_like(grid_matrix, dtype=np.int32)
    for idx in current_selection:
        scaled = [x_scale(current_emb[idx][0]), y_scale(current_emb[idx][1])]
        grid_distribution[scaled[0], scaled[1]] += 1


    ## generate grid vertex value info
    ## 현재 naive한 삼각형 커널
    for i in range(grid_size):
        for j in range(grid_size):
            value = grid_distribution[i][j]
            kernel_size = 2
            for ii in range(-kernel_size, kernel_size + 1):
                for jj in range(-kernel_size, kernel_size + 1):
                    
                    dist = abs(ii) + abs(jj)
                    if (dist > value) or (i + ii >= grid_size) or (j + jj >= grid_size):
                        continue 
                    current_kernel_density = value - dist * (value / kernel_size)
                    grid_matrix[i + ii, j + jj] += current_kernel_density
    
    ## currently isovalue : 20% point of the max value of the grid matrix
    return np.max(grid_matrix) * 0.2



## marching squares algorithm
def marching_squares(grid_matrix, grid_sign, grid_square, contour_configuration, isovalue):

    grid_size = grid_matrix.shape[0]

    ## grid matrix to grid sign
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_matrix[i, j] >= isovalue:
                grid_sign[i, j] = 1
            else:
                grid_sign[i, j] = 0
    
    contour_square = []
    ## grid sign to grid square
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            ## simple loop unrolling
            index_3 = grid_sign[i, j]
            index_2 = grid_sign[i + 1, j]
            index_1 = grid_sign[i, j + 1]
            index_0 = grid_sign[i + 1, j + 1]

            index_fi = 8 * index_3 + 4 * index_2 + 2 * index_1 + index_0
            grid_square[i, j] =  index_fi
            if ( 0 < index_fi < 15):
                contour_square.append((i, j))
            

    ## linear interpolation
    contours = []
    for (i, j) in contour_square:
        edge_direction = contour_configuration[int(grid_square[i, j])]
        for edge in edge_direction:
            ## unrolling
            edge_start = edge[0]
            coor_p = np.array([i + edge_start[0] % 2, j + edge_start[0] // 2])
            coor_q = np.array([i + edge_start[1] % 2, j + edge_start[1] // 2])
            s_p = grid_matrix[coor_p[0], coor_p[1]]
            s_q = grid_matrix[coor_q[0], coor_p[1]]
            alpha = (isovalue - s_p) / (s_q - s_p)
            edge_start_coor = coor_p * (1 - alpha) + coor_q * alpha
            
            edge_end = edge[0]
            coor_p = np.array([i + edge_end[0] % 2, j + edge_end[0] // 2])
            coor_q = np.array([i + edge_end[1] % 2, j + edge_end[1] // 2])
            s_p = grid_matrix[coor_p[0], coor_p[1]]
            s_q = grid_matrix[coor_q[0], coor_p[1]]
            alpha = (isovalue - s_p) / (s_q - s_p)
            edge_end_coor = coor_p * (1 - alpha) + coor_q * alpha

            edge_coor = [edge_start_coor, edge_end_coor]
            contours.append(edge_coor)

            
    


