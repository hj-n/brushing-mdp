'''
The code for generating the contour of current selection

kernel_density: performs 2d histogram-based approximated kde (HAKDE)
marching_squares: perform marching squares algorithm for final contour computation

'''

import numpy as np 
import math
import copy

from numba import cuda


def generate(current_selection, current_emb, max_emb, min_emb):

    g_size = 25
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
    contours = marching_squares(grid_matrix, grid_sign, grid_square, contour_configuration, isovalue)

    return contours

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
            index_0 = grid_sign[i, j]
            index_1 = grid_sign[i + 1, j]
            index_2 = grid_sign[i, j + 1]
            index_3 = grid_sign[i + 1, j + 1]

            index_fi = 8 * index_3 + 4 * index_2 + 2 * index_1 + index_0
            grid_square[i, j] =  index_fi
            if ( 0 < index_fi < 15):
                contour_square.append((i, j))

    ## Polygon Skeleton Extraction
    ### mark valid / visited points for fast search
    grid_mark = np.zeros_like(grid_square)

    edge_directions = []
    edge_positions = {}

    idx = 0
    for (i, j) in contour_square:
        edge_direction = contour_configuration[int(grid_square[i, j])]
        
    
        for single_edge_direction in edge_direction:
            new_direction = [
                [
                    [i + single_edge_direction[0][0]  % 2, j + single_edge_direction[0][0] // 2],
                    [i + single_edge_direction[0][1]  % 2, j + single_edge_direction[0][1] // 2]
                ],
                [
                    [i + single_edge_direction[1][0]  % 2, j + single_edge_direction[1][0] // 2],
                    [i + single_edge_direction[1][1]  % 2, j + single_edge_direction[1][1] // 2]
                ],
            ]

            grid_mark[i, j] += 1
            edge_directions.append(new_direction)
            
            key = str(i) + "_" + str(j)
            if key in edge_positions:
                edge_positions[key].append(idx)
            else:
                edge_positions[key] = [idx]

            idx += 1
            
    
    polygon = copy.deepcopy(edge_directions[0])

    def check_next_point(position_x, position_y, edge_positions, edge_directions, current_end, current_coor, polygon):
        key = str(position_x) + "_" + str(position_y)
        is_finished = False
        for (real_idx, idx) in enumerate(edge_positions[key]):
            edge = edge_directions[idx]
            if (edge[0][0][0] ==  current_end[0][0] and edge[0][0][1] ==  current_end[0][1] and 
                edge[0][1][0] ==  current_end[1][0] and edge[0][1][1] ==  current_end[1][1]):
                grid_mark[position_x, position_y] = -1 ## mark as visited
                polygon.append(edge[1])
                edge_positions[key].pop(real_idx)
                is_finished = True
                break
        return is_finished

    grid_mark[contour_square[0][0], contour_square[0][1]] -= 1 ## mark as visited
    current_coor = contour_square[0]
    for i in range(1, len(edge_directions) - 1):  
        current_end = polygon[-1]
        
        if current_coor[0] >= 1 and grid_mark[current_coor[0] - 1, current_coor[1]] > 0: ## can go left
            is_finished = check_next_point(current_coor[0] - 1, current_coor[1], 
                                           edge_positions, edge_directions, 
                                           current_end, current_coor, polygon)
            if (is_finished):
                current_coor = (current_coor[0] - 1, current_coor[1])
                continue
        if current_coor[0] <= (grid_size - 3) and grid_mark[current_coor[0] + 1, current_coor[1]] > 0: ## can go right
            is_finished = check_next_point(current_coor[0] + 1, current_coor[1], 
                                           edge_positions, edge_directions, 
                                           current_end, current_coor, polygon)
            if (is_finished):
                current_coor = (current_coor[0] + 1, current_coor[1])
                continue
        if current_coor[1] >= 1 and grid_mark[current_coor[0], current_coor[1] - 1] > 0: ## can go up
            is_finished = check_next_point(current_coor[0] , current_coor[1] - 1, 
                                           edge_positions, edge_directions, 
                                           current_end, current_coor, polygon)
            if (is_finished):
                current_coor = (current_coor[0], current_coor[1] - 1)
                continue
        if current_coor[1] <= (grid_size - 3) and grid_mark[current_coor[0], current_coor[1] + 1] > 0: ## can go down
            is_finished = check_next_point(current_coor[0], current_coor[1] + 1, 
                                           edge_positions, edge_directions, 
                                           current_end, current_coor, polygon)
            if (is_finished):
                current_coor = (current_coor[0], current_coor[1] + 1)
                continue
        
    ## Conver Polygon Skeleton into real polygon coordinates
    polygon_coor = []
    for point in polygon:
        s_p = grid_matrix[point[0][0], point[0][1]]
        s_q = grid_matrix[point[1][0], point[1][1]]
        alpha = (isovalue - s_p) / (s_q - s_p)
        coor = [point[0][0] * (1 - alpha) + point[1][0] * alpha, point[0][1] * (1 - alpha) + point[1][1] * alpha]
        polygon_coor.append(coor)
    


    return polygon_coor
    


