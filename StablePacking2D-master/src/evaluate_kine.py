
from kine_2d import ContPoint, Element,ContType
import numpy as np
from kine_2d import cal_Aglobal, solve_force_rigid

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import skimage.segmentation as segmentation


left_bound_pixel_value = 254
right_bound_pixel_value = 253
base_pixel_value = 255

class Parameter:

    # kinematics
    boundary = "tilting-table"
    stress = 30/100
    mu = 0.3
    fc = 0
    cohesion = 0
    density = 18
    calibrate_head_joint = False
    ignore_head_joint = False
    min_contp_distance = 1000
    offset = 1
    contact_detection_method = "uniform"
    ctname = "friction"


parameters = Parameter()
parameters_head = Parameter()
parameters_mortar = Parameter()
_conttype = ContType("friction", parameters)


def plot_model(elems,contps):
    for element in elems.values():
        if element.type == 'stone':
            plt.scatter(element.center[0],element.center[1],c = 'r',s = 100)
        else:
            plt.scatter(element.center[0],element.center[1],c = 'b',s = 100)
    for contp in contps.values():
        plt.scatter(contp.coor[0],contp.coor[1],c = 'g',s = 100)
    plt.show()


def initialize_kine_model(wall_id_matrix):
    elems = dict()
    contps = dict()
    #find the center of stone
    center_ground = np.mean(np.argwhere(wall_id_matrix==base_pixel_value),axis = 0)
    # find the mass of stone
    mass  = np.sum(np.where(wall_id_matrix==base_pixel_value,1,0))
    # add 1 to center y of ground to be sure that the center of the ground is below the contact point with stone, in case when ground is 1 pixel thick
    elems[base_pixel_value] = Element(
                base_pixel_value, [center_ground[1],center_ground[0]+1], mass, None, type='ground')
    
    # # add bound as fixed
    # #find the center of stone
    # center_ground = np.mean(np.argwhere(wall_id_matrix==left_bound_pixel_value),axis = 0)
    # # find the mass of stone
    # mass  = np.sum(np.where(wall_id_matrix==left_bound_pixel_value,1,0))
    # # add 1 to center y of ground to be sure that the center of the ground is below the contact point with stone, in case when ground is 1 pixel thick
    # elems[left_bound_pixel_value] = Element(
    #             left_bound_pixel_value, [center_ground[1],center_ground[0]+1], mass, None, type='ground')
    
    # #find the center of stone
    # center_ground = np.mean(np.argwhere(wall_id_matrix==right_bound_pixel_value),axis = 0)
    # # find the mass of stone
    # mass  = np.sum(np.where(wall_id_matrix==right_bound_pixel_value,1,0))
    # # add 1 to center y of ground to be sure that the center of the ground is below the contact point with stone, in case when ground is 1 pixel thick
    # elems[right_bound_pixel_value] = Element(
    #             right_bound_pixel_value, [center_ground[1],center_ground[0]+1], mass, None, type='ground')
    return elems, contps




def update_kine_model(elems, contps,wall_id_matrix,stone_to_add):
    #ignore bound when calculating the ka
    wall_id_matrix = np.where((wall_id_matrix==left_bound_pixel_value)|(wall_id_matrix==right_bound_pixel_value),0,wall_id_matrix)
    #binarilize the stone
    stone_to_add = np.where(stone_to_add>0,1,0)
    # find the maximum id of the contps
    if len(contps) == 0:
        maxPointID = 0
    else:
        maxPointID = max(contps.keys())
    maxPointID += 1
    #find the maximum id of the elements in wall_id_matrix
    #maxElemID = np.max(np.where(wall_id_matrix!=base_pixel_value,wall_id_matrix,0))
    maxElemID =max(elems.keys())
    maxElemID += 1
    # find the outer boundary of the stone
    stone_boundary = segmentation.find_boundaries(stone_to_add,connectivity = 2,mode = 'outer',background = 0)
    #find the center of stone
    center_stone = np.mean(np.argwhere(stone_to_add),axis = 0)
    center_stone = np.flip(center_stone)
    # find the mass of stone
    mass  = np.sum(np.where(stone_to_add!=0,1,0))
    # find the inner boundary of the wall
    wall_binary_mask = np.where(wall_id_matrix>0,1,0)
    wall_boundary = segmentation.find_boundaries(wall_binary_mask,connectivity = 2,mode = 'inner',background = 0)
    # find the intersection of the two boundaries
    intersection_id = np.where(np.logical_and(stone_boundary,wall_boundary),wall_id_matrix,0)
    # for each unique id in the intersection, create one contact point
    for id in np.unique(intersection_id):
        if id == 0:
            continue
        # find the coordinates of the contact point
        coords = np.argwhere(intersection_id == id)
        coords = np.flip(coords,axis = 1)
        # find the center of the contact point
        center = np.mean(coords,axis = 0)
        # find the tangent orientation of the contact point
        if coords.shape[0] == 1:
            # the orientation is the orientation of neighboring pixels
            bound_coords = np.flip(np.argwhere(wall_boundary),axis = 1)
            neighbor_coords = bound_coords[(np.linalg.norm(bound_coords-coords[0],axis = 1))<5]
            orientation = np.mean(np.diff(neighbor_coords,axis = 0),axis = 0)
        else:
            orientation = np.mean(np.diff(coords,axis = 0),axis = 0)
        orientation = orientation/np.linalg.norm(orientation)
        # find the normal of the contact point
        normal = np.asarray([orientation[1],-orientation[0]])
        # orient the normal such that it points to the center of the stone
        center_to_point_vector = -center + center_stone
        if np.dot(normal,center_to_point_vector) < 0:
            normal = -normal
        # sor the contact points
        projections = np.matmul(
                        coords, orientation)
        coordinate_sort_indices = np.argsort(
                        projections.reshape(1, -1))[0]
        coordinate_end1 = coords[coordinate_sort_indices[0], :]
        coordinate_end2 = coords[coordinate_sort_indices[-1], :]
        # create the contact point for the stone
        contps[maxPointID] = ContPoint(maxPointID, [
            coordinate_end1[0], coordinate_end1[1]], maxElemID, id, orientation.tolist(), [normal[0],normal[1]], _conttype)
        maxPointID += 1
        contps[maxPointID] = ContPoint(maxPointID, [
            coordinate_end2[0], coordinate_end2[1]], maxElemID, id, orientation.tolist(), [normal[0],normal[1]], _conttype)
        maxPointID += 1
        # create the contact point
        contps[maxPointID] = ContPoint(maxPointID, [
            coordinate_end1[0], coordinate_end1[1]], id, maxElemID, orientation.tolist(), [-normal[0],-normal[1]], _conttype)
        maxPointID += 1
        contps[maxPointID] = ContPoint(maxPointID, [
            coordinate_end2[0], coordinate_end2[1]], id, maxElemID, orientation.tolist(), [-normal[0],-normal[1]], _conttype)
        maxPointID += 1
        
    # create the element
    elems[maxElemID] = Element(
                maxElemID, center_stone, mass, None, type='stone')
    return elems, contps


def evaluate_kine(elems, contps):
    """Evaluate the limit tilting angle of the placed stones

    :param final_base: The base to be evaluated
    :type final_base: Stonepacker2D.base.Base
    :return: The limit tilting angle of the placed stones, minimal of two direction
    :rtype: float
    """
    for element in elems.values():
        element.dl = [0, element.mass, 0]
        element.ll = [element.mass, 0, 0]

    Aglobal = cal_Aglobal(elems, contps)

    one_direction_solution = solve_force_rigid(elems, contps, Aglobal)
    one_direction = one_direction_solution['limit_force']
    for element in elems.values():
        element.dl = [0, element.mass, 0]
        element.ll = [-element.mass, 0, 0]

    another_direction_solution = solve_force_rigid(elems, contps, Aglobal)
    another_direction = another_direction_solution['limit_force']
    return round(min(one_direction, another_direction), 2)

