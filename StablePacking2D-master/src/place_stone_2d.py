import cv2
import numpy as np
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from scipy import ndimage
from rotate_stone import rotate
from evaluate_kine import initialize_kine_model, update_kine_model, evaluate_kine,plot_model
from matplotlib import cm
from matplotlib.colors import ListedColormap
import time
from multiprocessing import Pool
import scipy.ndimage
from skimage.morphology import flood
import math
base_pixel_value = 255
left_bound_pixel_value = 254
right_bound_pixel_value = 253
NUMBER_OF_PIXELS_AS_EPSILON = 3#5
INTERLOCKING_PORTION = 10/4
INTERLOCKING_PIXEL = 3#even the value can be smaller, remember the "touching" criteria that will determine the interlocking of the next stone


def get_phi_distance(base):
    # # compute the minimal distance from each pixel of the stone to non-zero pixels of base
    # phi = np.zeros_like(base)
    # for i in range(base.shape[0]):
    #     for j in range(base.shape[1]):
    #         if base[i, j] == 0:
    #             phi[i, j] = np.min(np.linalg.norm(
    #                 np.argwhere(base)-np.array([i, j]), axis=1))
    #         else:
    #             continue
    # compute the minimal distance
    base_reverted  = np.where(base!=0,0,1)
    new_phi = scipy.ndimage.distance_transform_edt(base_reverted, return_distances=True,
                                         return_indices=False)
    return new_phi


def get_proximity_metric(base_phi, brick_bounding_box_matrix_mirrow):
    shift_to = [(brick_bounding_box_matrix_mirrow.shape[0]-1) // 2,
               (brick_bounding_box_matrix_mirrow.shape[1]-1) // 2]
    proximity_metric = ndimage.convolve(
       base_phi, brick_bounding_box_matrix_mirrow, mode='constant', cval=1.0, origin=shift_to)
    #proximity_metric = np.multiply(base_phi, stone)
    return proximity_metric

def get_height(matrix):
    # matrix where the last row in dimension 0 is 0, the others are 1
    height_matrix = np.ones_like(matrix)
    height_matrix[-1,:] = 0
    height_matrix = scipy.ndimage.distance_transform_edt(height_matrix, return_distances=True, return_indices=False)
    # normalize
    height_matrix = height_matrix/np.max(height_matrix)
    # cubic
    height_matrix = height_matrix
    return height_matrix

def get_distance_to_interlocking(matrix, with_bound = False):
    matrix = np.where(matrix==base_pixel_value,0,matrix)
    if not with_bound:
        matrix = np.where((matrix==left_bound_pixel_value)|(matrix==right_bound_pixel_value),0,matrix)
    
   
    # compute gradient
    sobelx = cv2.Sobel(matrix,cv2.CV_64F,1,0,ksize=3)
    abs_sobel64f = np.absolute(sobelx)
    
    # plt.imshow(abs_sobel64f)
    # plt.show()
    matrix_boundary = np.where(abs_sobel64f>0,1,0)
    # plt.imshow(matrix_boundary)
    # plt.show()
    # erosion and dilation (cv2 closing not working as expected)
    kernel = np.ones((NUMBER_OF_PIXELS_AS_EPSILON*2,1),np.uint8)
    matrix_boundary = cv2.erode(matrix_boundary.astype('uint8'),kernel,iterations = 1)
    matrix_boundary = cv2.dilate(matrix_boundary.astype('uint8'),kernel,iterations = 1)
    # if with_bound:
    #     kernel = np.ones((NUMBER_OF_PIXELS_AS_EPSILON,1),np.uint8)
    #     matrix_boundary = cv2.erode(matrix_boundary.astype('uint8'),kernel,iterations = 1)
    if not with_bound:
        matrix_boundary = cv2.dilate(matrix_boundary.astype('uint8'),kernel,iterations = 1)
    
    # revert 0 and 1
    matrix_boundary = np.where(matrix_boundary==0,1,0)
   
    # check if there is any zero pixel
    if len(np.argwhere(matrix_boundary==0))==0:
        return np.ones_like(matrix_boundary)*0.12
    # compute the distance to the boundary
    distance_matrix = scipy.ndimage.distance_transform_edt(matrix_boundary, return_distances=True, return_indices=False)
    return distance_matrix


def add_stone(wall, wall_seg_matrix,stone,p=None,relaxed_mason_criteria=False):
    """ find the best position of a given stone pose
    """
    # compute kernel for convolution
    brick_bounding_box_matrix = regionprops(stone.astype(np.uint8))[0].image
    brick_bounding_box_matrix_mirrow = np.flip(
        np.flip(brick_bounding_box_matrix, axis=0), axis=1)
    shift_to = [(brick_bounding_box_matrix_mirrow.shape[0]-1) // 2,
                (brick_bounding_box_matrix_mirrow.shape[1]-1) // 2]
    
    # ------------------ Criteria 1: non overlapping mask
    non_overlap_mask = np.where(ndimage.convolve(
        wall, brick_bounding_box_matrix_mirrow, mode='constant', cval=1.0, origin=shift_to) == 0, 1, 0)
    
    # ------------------ Criteria 2: the bottom face of the stone should be in contact with the wall
    kernel_dilation_y = np.ones((3, 1), np.uint8)
    contour_up = cv2.dilate(wall, kernel_dilation_y,
                            anchor=(0, 1), iterations=1)
    overlap_contour_mask = np.where(ndimage.convolve(
        contour_up, brick_bounding_box_matrix_mirrow, mode='constant', cval=0.0, origin=shift_to) != 0, 1, 0)

    # ------------------ Criteria 3: the left part and right part (with respect to CoM) of the stone should both be in contact with the wall
    kernel_dilation = np.ones((3, 3), np.uint8)
    wall_with_no_bound = np.where((wall_seg_matrix!=left_bound_pixel_value)&(wall_seg_matrix!=right_bound_pixel_value),wall,0)
    contour_dilated = cv2.dilate(wall_with_no_bound, kernel_dilation,
                            anchor=(1, 1), iterations=1)
    # left contact
    stone_center = regionprops(stone.astype(np.uint8))[0].centroid
    left_mask = np.zeros_like(stone)
    left_mask[:,0:int(stone_center[1])] = 1
    left_brick_bounding_box_matrix_mirrow = brick_bounding_box_matrix_mirrow*left_mask
    left_overlap_contour_mask = np.where(ndimage.convolve(
        contour_dilated, left_brick_bounding_box_matrix_mirrow, mode='constant', cval=0.0, origin=shift_to) != 0, 1, 0)
    #right contact
    right_mask = np.zeros_like(stone)
    right_mask[:,int(stone_center[1])+1:] = 1
    right_brick_bounding_box_matrix_mirrow = brick_bounding_box_matrix_mirrow*right_mask
    right_overlap_contour_mask = np.where(ndimage.convolve(
        contour_dilated, right_brick_bounding_box_matrix_mirrow, mode='constant', cval=0.0, origin=shift_to) != 0, 1, 0)
    
    # feasible regions combined the three criteria
    # if relax contact criteria, only non overlapping criteria (Criteria 1) is considered
    # othersize, all three criteria are considered
    if relaxed_mason_criteria:
        region_potential = np.multiply(non_overlap_mask, overlap_contour_mask)
    else:
        region_potential = np.multiply(non_overlap_mask, left_overlap_contour_mask)
        region_potential = np.multiply(region_potential, right_overlap_contour_mask)
        region_potential = np.multiply(region_potential, overlap_contour_mask)
    if len(np.argwhere(region_potential!=0))==0:
        return None, {"interlocking":-np.inf,"optimization_score":-np.inf,"refinement_direction":None}
    

    
    # ------------------ Criteria 4: interlocking
    # it is a soft criteria: we relax it if necessary to make sure that there exist regions that matches this criteria
    stone_width = regionprops(stone.astype(np.uint8))[0].bbox[3]-regionprops(stone.astype(np.uint8))[0].bbox[1]
    stone_height = regionprops(stone.astype(np.uint8))[0].bbox[2]-regionprops(stone.astype(np.uint8))[0].bbox[0]
    # min_scale_stone = min(stone_width,stone_height)
    # max_scale_stone = max(stone_width,stone_height)

    #left interlocking distance map and convolve with the bounding of stone
    interlocking_distance = get_distance_to_interlocking(wall_seg_matrix)#maximize the distance to the interlocking
    brick_bounding_box_matrix_boundary_only = np.zeros_like(brick_bounding_box_matrix)#using bounding box instead of actual stone shape
    brick_bounding_box_matrix_boundary_only[-1,-1] = 1
    left_interlocking_distance_convolved = ndimage.minimum_filter(
        interlocking_distance, footprint=brick_bounding_box_matrix_boundary_only, mode='constant', cval=np.inf, origin=[-shift_to[0],-shift_to[1]])
    left_interlocking_distance_convolved[left_interlocking_distance_convolved==0.12] = np.inf
    #right interlocking distance map and convolve with the bounding of stone
    brick_bounding_box_matrix_boundary_only = np.zeros_like(brick_bounding_box_matrix)#using bounding box instead of actual stone shape
    brick_bounding_box_matrix_boundary_only[-1,0] = 1
    right_interlocking_distance_convolved = ndimage.minimum_filter(
        interlocking_distance, footprint=brick_bounding_box_matrix_boundary_only, mode='constant', cval=np.inf, origin=[-shift_to[0],-shift_to[1]])
    right_interlocking_distance_convolved[right_interlocking_distance_convolved==0.12] = np.inf
    #take the maximum of left and right interlocking distance
    interlocking_distance_convolved = np.minimum(left_interlocking_distance_convolved,right_interlocking_distance_convolved)

    touch_distance = get_distance_to_interlocking(wall_seg_matrix, with_bound=True)
    brick_bounding_box_matrix_boundary_only = np.zeros_like(brick_bounding_box_matrix)#using bounding box instead of actual stone shape
    #print("The stone center is at ",stone_center[0])
    #print("The stone center int is at ",int(stone_center[0]))
    brick_bounding_box_matrix_boundary_only[0:math.ceil(stone_center[0]),-1] = 1
    left_touch_distance_convolved = ndimage.minimum_filter(
        touch_distance, footprint=brick_bounding_box_matrix_boundary_only, mode='constant', cval=np.inf, origin=[-shift_to[0],-shift_to[1]])
    #right interlocking distance map and convolve with the bounding of stone
    brick_bounding_box_matrix_boundary_only = np.zeros_like(brick_bounding_box_matrix)#using bounding box instead of actual stone shape
    brick_bounding_box_matrix_boundary_only[0:math.ceil(stone_center[0]),0] = 1
    right_touch_distance_convolved = ndimage.minimum_filter(
        touch_distance, footprint=brick_bounding_box_matrix_boundary_only, mode='constant', cval=np.inf, origin=[-shift_to[0],-shift_to[1]])

    # #neighbor height distance map and convolve with the bounding of stone
    # neighbor_height_distance = get_distance_to_neighbor_height(wall_seg_matrix)#maximize the distance to the interlocking
    # #brick_bounding_box_matrix_mirrow_boundary_only = brick_bounding_box_matrix_mirrow.copy()
    # #brick_bounding_box_matrix_mirrow_boundary_only[:,1:-1] = 0
    # #brick_bounding_box_matrix_boundary_only = brick_bounding_box_matrix.copy()#minimum_filter does not mirrow the kernel
    # brick_bounding_box_matrix_boundary_only = np.ones_like(brick_bounding_box_matrix)
    # brick_bounding_box_matrix_boundary_only[1:,:] = 0
    # #cval= large value makes sure that when stone is placed on the boundary the minimal distance is compared with stones, not emptiness
    # neighbor_height_distance_convolved = ndimage.minimum_filter(
    #     neighbor_height_distance, footprint=brick_bounding_box_matrix_boundary_only, mode='constant', cval=10.0, origin=[-shift_to[0],-shift_to[1]])

    # #distance to bound
    # brick_bounding_box_matrix_boundary_only = np.ones_like(brick_bounding_box_matrix)#using bounding box instead of actual stone shape
    # brick_bounding_box_matrix_boundary_only[:,1:-1] = 0
    # distance_to_bound = ndimage.minimum_filter(
    #     bound_distance, footprint=brick_bounding_box_matrix_boundary_only, mode='constant', cval=0.0, origin=[-shift_to[0],-shift_to[1]])
    # distance_to_bound = distance_to_bound*wall.shape[1]/2

    # threshold of interlocking length
    interlocking_thre = min(INTERLOCKING_PIXEL,stone_width/2)
    interlocking_thre = min(interlocking_thre,stone_height)
    # mason_criteria = np.where(((interlocking_distance_convolved>=interlocking_thre) | (interlocking_distance_convolved==0.12))\
    #                           ,1,0)
    # mason_criteria = np.where(((left_interlocking_distance_convolved>=interlocking_thre)|(left_touch_distance_convolved<=1))&\
    # ((right_interlocking_distance_convolved>=interlocking_thre)|(right_touch_distance_convolved<=1)),1,0)

    mason_criteria = np.where(((left_interlocking_distance_convolved>=interlocking_thre)&(right_touch_distance_convolved<=1))|\
    ((right_interlocking_distance_convolved>=interlocking_thre)&(left_touch_distance_convolved<=1))|\
        ((right_touch_distance_convolved<=0)&(left_touch_distance_convolved<=1)),1,0)

    # # plt.imshow(mason_criteria)
    # # plt.show()
    # mason_criteria2 = np.where(one_side_touch_distance_convolved<=1,1,0)#oneside touch
    # mason_criteria3 = np.where(two_side_touch_distance_convolved<=1,1,0)#two side touch
    # mason_criteria = np.where(((mason_criteria!=0)&(mason_criteria2!=0))|mason_criteria3!=0,1,0)
    iteration_ = 1
    while len(np.argwhere(np.multiply(region_potential,mason_criteria)!=0))==0 and iteration_<4:
        # mason_criteria = np.where(((interlocking_distance_convolved>=(interlocking_thre/(iteration_+1))) | (interlocking_distance_convolved==0.12))\
        #                       ,1,0)
        # mason_criteria = np.where(((left_interlocking_distance_convolved>=interlocking_thre/(iteration_+1))|(left_touch_distance_convolved<=1))&\
        #     ((right_interlocking_distance_convolved>=interlocking_thre/(iteration_+1))|(right_touch_distance_convolved<=1)),1,0)
        # #mason_criteria = np.multiply(mason_criteria,mason_criteria2)
        # mason_criteria = np.where(((mason_criteria!=0)&(mason_criteria2!=0))|mason_criteria3!=0,1,0)
        mason_criteria = np.where(((left_interlocking_distance_convolved>=interlocking_thre/(iteration_+1))&(right_touch_distance_convolved<=1))|\
        ((right_interlocking_distance_convolved>=interlocking_thre/(iteration_+1))&(left_touch_distance_convolved<=1))|\
        ((right_touch_distance_convolved<=1)&(left_touch_distance_convolved<=1)),1,0)
        iteration_+=1
    # plt.imshow(mason_criteria)
    # plt.show()
    #mason_criteria = np.where(((interlocking_distance_convolved<=1) | (interlocking_distance_convolved==0.12)),1,0)
    #mason_criteria = np.ones_like(region_potential)
    if relaxed_mason_criteria:
    #if relaxed_mason_criteria or len(np.argwhere(np.multiply(region_potential,mason_criteria)!=0))==0:
        mason_criteria = np.ones_like(region_potential)
    region_potential = np.multiply(region_potential,mason_criteria)
    
    # ------------------ Optimization Objective 1: minimize distance to exisitng walls
    # proximity map
    base_phi = get_phi_distance(wall)

    # ------------------ Optimization Objective 2: minimize distance to ground
    # height map
    base_height = get_height(wall)

    # ------------------ Combine the two optimization objectives
    # find the location that minimize the proximity_metric nad the height metric and is part of feasible regions
    if p is not None:
        weight_height = p*np.sqrt(stone_width**2+stone_height**2)*1000
    else:
        weight_height = np.sqrt(stone_width**2+stone_height**2)*1000
    score_optimization_map = -base_phi-weight_height*base_height
    score_optimization = get_proximity_metric(score_optimization_map, brick_bounding_box_matrix_mirrow)
    score_optimization = score_optimization
    score_potential = np.where(region_potential != 0, score_optimization, -np.inf)
    best_score = np.max(score_potential)
    best_loc = np.argwhere(score_potential == best_score)[-1]#!should choose -1 instead of 0, otherwise position (0,0) will always be chosen

    # ------------------Output features of the best location
    # get the interlocking distance of the best location
    best_interlocking_distance = interlocking_distance_convolved[best_loc[0],best_loc[1]]
    # get the neighbor height distance of the best location
    #best_neighbor_height_distance = neighbor_height_distance_convolved[best_loc[0],best_loc[1]]
    # get the proximity metric of the best location
    best_score_optimization = score_optimization[best_loc[0],best_loc[1]]

    #distance_to_bound = distance_to_bound[best_loc[0],best_loc[1]]

    # get the refinement direction
    score_optimization_gradient = np.gradient(score_optimization)
    # get the direction of refinement for the best location
    best_refinement_direction = [np.sign(score_optimization_gradient[0][best_loc[0],best_loc[1]]),
                                 np.sign(score_optimization_gradient[1][best_loc[0],best_loc[1]])]   
    
    return best_loc, {"interlocking":best_interlocking_distance,"optimization_score":best_score_optimization,"refinement_direction":best_refinement_direction}

def transform(stone, location):
    shape = stone.shape
    transformed_stone = np.zeros_like(stone)
    transformed_stone[int(location[0]):, int(
        location[1]):] = stone[:shape[0]-int(location[0]), : shape[1]-int(location[1])]
    return transformed_stone


def get_best_placement(wall, wall_seg_matrix,stone, rotation_angle_options,elems = {}, contps = {}, weight_height = 1,func = add_stone, rotation_function = rotate,nb_processor = 1,relaxed_mason_criteria = False,iteration = 0,result_dir = None):
    """Find the best placement of a stone considering all possible rotation poses
    """
    # variables storing the best placement
    best_rotate_pose_index = 0
    best_score = -np.inf
    best_loc = None
    best_direction = [None,None]
    # For stone, find the best placement of each rotated pose, eliminate unstable poses, choose the best from the stable ones
    if nb_processor == 1:# sequential search each rotation pose
        for rotate_pose_index,rotate_sequence in enumerate(rotation_angle_options):
            #rotate stone
            stone_,_ = rotation_function(stone,rotate_sequence)
            #find the position
            best_loc_this_pose, evaluation_this_pose= func(
                wall, wall_seg_matrix,stone_,p=weight_height,relaxed_mason_criteria=relaxed_mason_criteria)
            best_score_this_pose = evaluation_this_pose["optimization_score"]
            print("Score with rotation {} is {}".format(rotate_sequence,best_score_this_pose))
            # skip the current rotation pose if no feasible position is found
            if best_score_this_pose == -np.inf:
                continue
            best_direction_this_pose = evaluation_this_pose["refinement_direction"]
            
            # check stability of the best position
            # transform stone to the best position
            stone_to_add = stone_
            stone_to_add_wall_size = np.zeros_like(wall)
            stone_to_add_wall_size[0:stone_to_add.shape[0],0:stone_to_add.shape[1]] = stone_to_add
            transformed_stone = transform(stone_to_add_wall_size, best_loc_this_pose)
            #KA analysis
            print("BUILDING KA MODEL")
            elems_ = elems.copy()
            contps_ = contps.copy()
            elems_, contps_ = update_kine_model(elems_, contps_,wall_seg_matrix,transformed_stone)
            print("EVALUATING KA MODEL")
            la_result = evaluate_kine(elems_, contps_)
            
            if la_result<=0:
                print("Not stable with ka {}".format(la_result))
                best_score_this_pose = -np.inf
            
            if  best_score_this_pose> best_score:
                best_rotate_pose_index = rotate_pose_index
                best_score = best_score_this_pose
                best_loc = best_loc_this_pose
                best_direction = best_direction_this_pose
    else:
        # parallel search
        for rotation_index_this_processor in range(0,len(rotation_angle_options),nb_processor):
            inputs = []
            for j in range(rotation_index_this_processor,min(rotation_index_this_processor+nb_processor,len(rotation_angle_options))):
                rotate_sequence = rotation_angle_options[j]
                stone_rot,_ = rotation_function(stone,rotate_sequence)
                input =(wall,wall_seg_matrix,stone_rot,weight_height,relaxed_mason_criteria)
                inputs.append(input)
            with Pool(nb_processor) as p:
                results = p.starmap(func, inputs)
            for j in range(len(results)):
                best_score_this_pose = results[j][1]["optimization_score"]
                best_loc_this_pose = results[j][0]
                print("Score with rotation {} is {}".format(rotation_angle_options[j+rotation_index_this_processor],best_score_this_pose))
                print("Position is {}".format(best_loc_this_pose))
                if best_score_this_pose == -np.inf:
                    continue
                
                stone_ = inputs[j][2]
        
                # check stability of the best position
                # transform stone to the best position
                stone_to_add = stone_
                stone_to_add_wall_size = np.zeros_like(wall)
                stone_to_add_wall_size[0:stone_to_add.shape[0],0:stone_to_add.shape[1]] = stone_to_add
                transformed_stone = transform(stone_to_add_wall_size, best_loc_this_pose)
                #KA analysis
                print("BUILDING KA MODEL")
                elems_ = elems.copy()
                contps_ = contps.copy()
                elems_, contps_ = update_kine_model(elems_, contps_,wall_seg_matrix,transformed_stone)
                print("EVALUATING KA MODEL")
                la_result = evaluate_kine(elems_, contps_)
                
                if la_result<=0:
                    best_score_this_pose = -np.inf              

                if  best_score_this_pose> best_score:
                    best_rotate_pose_index = rotation_index_this_processor+j
                    best_score = best_score_this_pose
                    best_loc = results[j][0]
                    best_direction = results[j][1]["refinement_direction"]
    return best_rotate_pose_index, best_score, best_loc,best_direction

def save_matrix(matrix,file_full_name,with_label = True):
    tab20_cm = cm.get_cmap('tab20')
    newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
    white = np.array([255/256, 255/256, 255/256, 1])
    newcolors[:1, :] = white
    newcmp = ListedColormap(newcolors)
    plt.imshow(matrix[:,:],cmap=newcmp,vmin=0,vmax = 255)

    # write region label at the center of each region
    if with_label:
        for region in regionprops(matrix.astype(np.uint8)):
            if region.label==base_pixel_value or region.label==left_bound_pixel_value or region.label==right_bound_pixel_value:
                continue
            y0, x0 = region.centroid
            plt.text(x0, y0, region.label, ha='center', va='center', color='black',fontsize=12)

    plt.axis('off')
    plt.savefig(file_full_name,dpi = 600,transparent=True)
    plt.close()
def generate_one_wall_best_pose_given_sequence(wall_i, result_dir=None, sequence=None, stones=None, wall_size=None,rotation_angle_options = None,weight_height = 1,nb_processor = 1, relaxed_mason_criteria = False,allowed_realxed_placement=0):
    """The main function to iterate given stones and place them with a predefined sequence

    :param wall_i: id of the wall
    :type wall_i: int
    :param result_dir: directory to store result, defaults to None
    :type result_dir: string
    :param sequence: the iterating order of the stones, defaults to None
    :type sequence: 1D numpy array
    :param stones: stone set, defaults to None
    :type stones: list of 2D numpy array
    :param wall_size: image size of wall, defaults to None
    :type wall_size: tuple of int
    :param rotation_angle_options: rotation poses to be considered for each stone, defaults to None
    :type rotation_angle_options: list of angles in degree
    :param weight_height: weight in optimization objective function, defaults to 1
    :type weight_height: int, optional
    :param nb_processor: number of cpus, defaults to 1
    :type nb_processor: int, optional
    :param relaxed_mason_criteria: whether using contact and interlocking criterias, defaults to False
    :type relaxed_mason_criteria: bool, optional
    :param allowed_realxed_placement: if relaxed_mason_criteria is true, the maximal number of stones placed, defaults to 0
    :type allowed_realxed_placement: int, optional
    :return: result containing built walls, unplaced stones, and stability verification ka model
    :rtype: dictionary
    """
    # initialize container for wall output
    wall = np.zeros((wall_size[0], wall_size[1]))
    wall_id_matrix = np.zeros((wall_size[0], wall_size[1]))
    stone_index_matrix = np.zeros((wall_size[0], wall_size[1]))
    transformation = np.zeros((len(sequence), 4))#d_x,d_y,angle,succesful
    # generate ground
    wall[-1, :] = 1
    wall_id_matrix[-1, :] = base_pixel_value
    stone_index_matrix[-1, :] = base_pixel_value
    wall[:,0] = 1
    wall_id_matrix[:,0] = left_bound_pixel_value
    stone_index_matrix[:,0] = left_bound_pixel_value
    wall[:,-1] = 1
    wall_id_matrix[:,-1] = right_bound_pixel_value
    stone_index_matrix[:,-1] = right_bound_pixel_value
    save_matrix(stone_index_matrix,result_dir+f'/wall_{wall_i}_ground.png',with_label=False)
    # initialize ka model
    elems, contps = initialize_kine_model(wall_id_matrix)
    #record time as iteration, number of rotation, search time, ka time
    recorded_time = np.zeros((len(sequence),4))
    # recap unplaced stones
    unplaced_stones = []
    # start iterating the stone sequence
    for i, stone_index in enumerate(sequence):
        save_matrix(stone_index_matrix,result_dir+f'/wall_{wall_i}_iteration_{i}.png',with_label=False)
        # time record and terminal output
        recorded_time[i,0] = i
        recorded_time[i,1] = len(rotation_angle_options)
        print("Iteration ",i,"/",len(sequence))
        print("Stone ",stone_index)
        # get the stone image
        stone_binary = stones[stone_index].copy()
        stone_binary[stone_binary > 0] = 1
        # optimize stone placement and rotation
        start_timer = time.time()
        nb_processor = min(nb_processor, len(rotation_angle_options))
        elems_for_placing = elems.copy()
        contps_for_placing = contps.copy()
        best_rotate_pose_index, best_score, best_loc,_ = get_best_placement(wall, wall_id_matrix,stone_binary,rotation_angle_options,elems = elems_for_placing,contps = contps_for_placing,weight_height = weight_height,func = add_stone,rotation_function = rotate,nb_processor = nb_processor,relaxed_mason_criteria = relaxed_mason_criteria)
        end_timer = time.time()
        recorded_time[i,2] = end_timer-start_timer
        
        # if feasible placement is found, add stone to wall
        # otherwise add stone to unplaced stones
        if best_score > -np.inf:
            # transform stone to the best position
            stone_to_add,translation = rotate(stones[stone_index],rotation_angle_options[best_rotate_pose_index])
            stone_to_add_wall_size = np.zeros_like(wall)
            stone_to_add_wall_size[0:stone_to_add.shape[0],0:stone_to_add.shape[1]] = stone_to_add
            transformed_stone = transform(stone_to_add_wall_size, best_loc)
            print("score of the placement: ",best_score)
            # save transformation
            transformation[i, 1] = best_loc[0]+translation[0]
            transformation[i, 0] = best_loc[1]+translation[1]
            transformation[i, 2] = rotation_angle_options[best_rotate_pose_index]
            transformation[i, 3] = 1
            # write image to file
            cv2.imwrite(result_dir+f'/wall_{wall_i}_iteration{i}_valid_stone_{stone_index}_best_pose_random_sequence.png',transformed_stone)
            save_matrix(transformed_stone,result_dir+f'/wall_{wall_i}_iteration{i}_valid_stone_{stone_index}_best_pose_random_sequence_id.png')
            #update wall
            wall += transformed_stone
            wall_id_matrix += (wall_i*100+i+1)*transformed_stone
            stone_index_matrix += (stone_index+1)*transformed_stone
            if relaxed_mason_criteria and i>=allowed_realxed_placement:
                # save all the rest stones as unplaced stones
                for stone_index in sequence[i+1:]:
                    unplaced_stones.append(stone_index)
                break
        else:
            print("No feasible placement")
            # add stone to unplaced stones
            unplaced_stones.append(stone_index)
    # write time record to txt file with column name
    np.savetxt(result_dir+f'/wall_{wall_i}_best_pose_given_sequence_time_record.txt',recorded_time,delimiter=',',header='iteration, nb_rotation, search_time, ka_time',comments='')

    # save image of the wall as figures
    cv2.imwrite(result_dir+f'/wall_{wall_i}_best_pose_random_sequence.png', wall)
    save_matrix(stone_index_matrix,result_dir+f'/wall_{wall_i}_best_pose_random_sequence_id.png')
    save_matrix(wall_id_matrix, result_dir+f'/wall_{wall_i}_with_sequence.png')
    return {"wall_id":wall_i, "unplaced_stones":unplaced_stones,"wall":wall,"wall_id_matrix":wall_id_matrix,"stone_index_matrix":stone_index_matrix,"elems":elems,"contps":contps,"transformation":transformation}

def generate_one_wall_best_pose_given_sequence_given_wall(wall_i,wall, wall_id_matrix,stone_index_matrix,elems,contps,result_dir=None, sequence=None, stones=None, wall_size=None,rotation_angle_options = None,weight_height = 1,nb_processor = 1,relaxed_mason_criteria = False,allowed_realxed_placement=0):
    """The main function to iterate given stones and place them on a half-built wall with a predefined sequence
    """
    transformation = np.zeros((len(sequence), 4))#d_x,d_y,angle,succesful
    #record time as iteration, number of rotation, search time, ka time
    recorded_time = np.zeros((len(sequence),4))
    # recap unplaced stones
    unplaced_stones = []
    # starting index
    starting_index = np.max(np.where((wall_id_matrix!=base_pixel_value)&(wall_id_matrix!=left_bound_pixel_value)&(wall_id_matrix!=right_bound_pixel_value),wall_id_matrix,0))
    # start iterating the stone sequence
    for i, stone_index in enumerate(sequence):
        # time record and terminal output
        recorded_time[i,0] = i
        recorded_time[i,1] = len(rotation_angle_options)
        print("Iteration ",i,"/",len(sequence))
        print("Stone ",stone_index)
        # get the stone image
        stone_binary = stones[stone_index].copy()
        stone_binary[stone_binary > 0] = 1
        # optimize stone placement and rotation
        start_timer = time.time()
        nb_processor = min(nb_processor, len(rotation_angle_options))
        elems_for_placing = elems.copy()
        contps_for_placing = contps.copy()
        best_rotate_pose_index, best_score, best_loc,_ = get_best_placement(wall, wall_id_matrix,stone_binary,rotation_angle_options,elems = elems_for_placing,contps = contps_for_placing,weight_height = weight_height,func = add_stone,rotation_function = rotate,nb_processor = nb_processor,relaxed_mason_criteria = relaxed_mason_criteria)
        end_timer = time.time()
        recorded_time[i,2] = end_timer-start_timer
        
        #check optimization feasibility
        if best_score > -np.inf:
            # transform stone to the best position
            stone_to_add,translation = rotate(stones[stone_index],rotation_angle_options[best_rotate_pose_index])
            stone_to_add_wall_size = np.zeros_like(wall)
            stone_to_add_wall_size[0:stone_to_add.shape[0],0:stone_to_add.shape[1]] = stone_to_add
            transformed_stone = transform(stone_to_add_wall_size, best_loc)
            print("score of the placement: ",best_score)
            # save transformation
            transformation[i, 1] = best_loc[0]+translation[0]
            transformation[i, 0] = best_loc[1]+translation[1]
            transformation[i, 2] = rotation_angle_options[best_rotate_pose_index]
            transformation[i, 3] = 1
            # write image to file
            cv2.imwrite(result_dir+f'/wall_{wall_i}_iteration{i}_valid_stone_{stone_index}_best_pose_random_sequence.png',transformed_stone)
            save_matrix(transformed_stone,result_dir+f'/wall_{wall_i}_iteration{i}_valid_stone_{stone_index}_best_pose_random_sequence_id.png')
            #update wall
            wall += transformed_stone
            wall_id_matrix += (starting_index+i+1)*transformed_stone
            stone_index_matrix += (stone_index+1)*transformed_stone
            if relaxed_mason_criteria and i>=allowed_realxed_placement:
                # save all the rest stones as unplaced stones
                for stone_index in sequence[i+1:]:
                    unplaced_stones.append(stone_index)
                break
        else:
            print("No feasible placement")
            # add stone to unplaced stones
            unplaced_stones.append(stone_index)
    # write time record to txt file with column name
    np.savetxt(result_dir+f'/wall_{wall_i}_best_pose_given_sequence_time_record.txt',recorded_time,delimiter=',',header='iteration, nb_rotation, search_time, ka_time',comments='')

    # save image of the wall as figures
    cv2.imwrite(result_dir+f'/wall_{wall_i}_best_pose_random_sequence.png', wall)
    save_matrix(stone_index_matrix,result_dir+f'/wall_{wall_i}_best_pose_random_sequence_id.png')
    save_matrix(wall_id_matrix, result_dir+f'/wall_{wall_i}_with_sequence.png')
    return {"wall_id":wall_i, "unplaced_stones":unplaced_stones,"wall":wall,"wall_id_matrix":wall_id_matrix,"stone_index_matrix":stone_index_matrix,"elems":elems,"contps":contps,"transformation":transformation}
    
