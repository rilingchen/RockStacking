import os
import numpy as np
from place_stone_2d import generate_one_wall_best_pose_given_sequence,generate_one_wall_best_pose_given_sequence_given_wall
from evaluate_kine import evaluate_kine
import cv2
import datetime
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from rotate_stone import rotate_axis_align

def read_example5_stones(scale):
    data_dir = '../data/example5/stones'
    stones = []
    read_sequence = []
    initial_angles = []
    for root, dir, files in os.walk(data_dir):
        for file in files:
            stone_img = cv2.imread(root+'/'+file, cv2.IMREAD_GRAYSCALE)
            # scale the stone
            stone_img = cv2.resize(stone_img, (0,0), fx=scale, fy=scale)
            #stone white background 0
            stone_img = np.where(stone_img <255, 1, 0)
            # if image is all zero, skip
            if np.sum(stone_img) == 0:
                continue
            # rotate the stone
            stone_img,initial_angle = rotate_axis_align(stone_img)
            initial_angles.append(initial_angle)
            stones.append(stone_img.astype('uint8'))
            # get the id of the stone
            stone_id = int(file.split('rubble_')[1].split('_')[0])
            read_sequence.append(stone_id)
    return stones,read_sequence,initial_angles



def generate_example5_best_pose_one_weight_one_size(scale,wall_size,weight = 1,seed_number = 0):
    """ An iterative construction process that 
    1. stack stone sets first by masons criteria 
    2. continue stacking without masons criteria if there are still stones left
    """
    #get the time date stamp
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = f'../data/example5/walls/wall_{time_stamp}'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    # write transformation to txt file
    with open(result_dir+'/transformation.txt', 'w+') as f:
        f.write('id;d_x;d_y;angle\n')

    wall_size = (int(wall_size[0]*scale),int(wall_size[1]*scale))
    stones,read_sequence, initial_angles = read_example5_stones(scale = scale)
    nb_processor = 4
    sequence = np.arange(len(stones))
    
    # randomly shuffle the sequence
    np.random.seed(seed_number)
    np.random.shuffle(sequence)
    

    rotation_angle_options = [0,90,180,270]
    #rotation_angle_options = [0]
    unused_number_pre = 0
    result = generate_one_wall_best_pose_given_sequence(0, result_dir, sequence, stones, wall_size, rotation_angle_options=rotation_angle_options,weight_height = weight,nb_processor = nb_processor)
    # write transformation to txt file
    with open(result_dir+'/transformation.txt', 'a') as f:
        for i_seq,which_stone in enumerate(sequence):
            if result['transformation'][i_seq][3]==0:
                continue
            stone_bbox = regionprops(stones[which_stone].astype(np.uint8))[0].bbox
            stone_center = regionprops(stones[which_stone].astype(np.uint8))[0].centroid
            stone_center = [stones[which_stone].shape[0]/2,stones[which_stone].shape[1]/2]
            initial_d_x = (stone_bbox[3]-stone_bbox[1])/2
            initial_d_x = stone_bbox[1]
            initial_d_y = (stone_bbox[2]-stone_bbox[0])/2
            initial_d_y = stone_bbox[0]
            initial_d_x = 0
            initial_d_y = 0

            d_x = (result['transformation'][i_seq][0]-initial_d_x+stone_center[1])/scale
            d_y = (wall_size[0]-(result['transformation'][i_seq][1]-initial_d_y+stone_center[0]))/scale
            angle_optimized = result['transformation'][i_seq][2]
            angle_initial = initial_angles[which_stone]
            stone_id = read_sequence[which_stone]
            f.write(f'{stone_id};{d_x};{d_y};{-angle_optimized-angle_initial}\n')
            
    max_iteration = 3
    try_iteraiton = 1
    while result['unplaced_stones'] != [] and try_iteraiton < max_iteration:
        if len(result['unplaced_stones'])== unused_number_pre:
            print("!! Relaxed Mason Criteria !!")
            unused_number_pre = len(result['unplaced_stones'])
            result_post = generate_one_wall_best_pose_given_sequence_given_wall(try_iteraiton,result['wall'],result['wall_id_matrix'],result['stone_index_matrix'],result['elems'],result['contps'], result_dir, result['unplaced_stones'], stones, wall_size, rotation_angle_options=rotation_angle_options,weight_height = weight,nb_processor = nb_processor,relaxed_mason_criteria=True)
            try_iteraiton+=1
            # write transformation to txt file
            with open(result_dir+'/transformation.txt', 'a') as f:
                for i_seq,which_stone in enumerate(result['unplaced_stones']):
                    if result_post['transformation'][i_seq][3]==0:
                        continue
                    stone_bbox = regionprops(stones[which_stone].astype(np.uint8))[0].bbox
                    stone_center = regionprops(stones[which_stone].astype(np.uint8))[0].centroid
                    stone_center = [stones[which_stone].shape[0]/2,stones[which_stone].shape[1]/2]
                    initial_d_x = (stone_bbox[3]-stone_bbox[1])/2
                    initial_d_x = stone_bbox[1]
                    initial_d_y = (stone_bbox[2]-stone_bbox[0])/2
                    initial_d_y = stone_bbox[0]
                    initial_d_x = 0
                    initial_d_y = 0
                    d_x = (result_post['transformation'][i_seq][0]-initial_d_x+stone_center[1])/scale
                    d_y = (wall_size[0]-(result_post['transformation'][i_seq][1]-initial_d_y+stone_center[0]))/scale
                    angle_optimized = result_post['transformation'][i_seq][2]
                    angle_initial = initial_angles[which_stone]
                    stone_id = read_sequence[which_stone]
                    f.write(f'{stone_id};{d_x};{d_y};{-angle_optimized-angle_initial}\n')
            result = result_post
        else:
            print("!! Resume Mason Criteria !!")
            unused_number_pre = len(result['unplaced_stones'])
            result_post = generate_one_wall_best_pose_given_sequence_given_wall(try_iteraiton,result['wall'],result['wall_id_matrix'],result['stone_index_matrix'],result['elems'],result['contps'], result_dir, result['unplaced_stones'], stones, wall_size, rotation_angle_options=rotation_angle_options,weight_height = weight,nb_processor = nb_processor)
            try_iteraiton+=1
            # write transformation to txt file
            with open(result_dir+'/transformation.txt', 'a') as f:
                for i_seq,which_stone in enumerate(result['unplaced_stones']):
                    if result_post['transformation'][i_seq][3]==0:
                        continue
                    stone_bbox = regionprops(stones[which_stone].astype(np.uint8))[0].bbox
                    stone_center = regionprops(stones[which_stone].astype(np.uint8))[0].centroid
                    stone_center = [stones[which_stone].shape[0]/2,stones[which_stone].shape[1]/2]
                    initial_d_x = (stone_bbox[3]-stone_bbox[1])/2
                    initial_d_x = stone_bbox[1]
                    initial_d_y = (stone_bbox[2]-stone_bbox[0])/2
                    initial_d_y = stone_bbox[0]
                    initial_d_x = 0
                    initial_d_y = 0
                    d_x = (result_post['transformation'][i_seq][0]-initial_d_x+stone_center[1])/scale
                    d_y = (wall_size[0]-(result_post['transformation'][i_seq][1]-initial_d_y+stone_center[0]))/scale
                    angle_optimized = result_post['transformation'][i_seq][2]
                    angle_initial = initial_angles[which_stone]
                    stone_id = read_sequence[which_stone]
                    f.write(f'{stone_id};{d_x};{d_y};{-angle_optimized-angle_initial}\n')
            result = result_post
    #evaluate kinematics
    KA_result = evaluate_kine(result['elems'], result['contps'])
    #evaluate void
    occupied_pixels = len(np.argwhere(result['wall'])!=0)
    total_pixels = result['wall'].shape[0]*result['wall'].shape[1]
    occupancy = occupied_pixels/total_pixels
    #write ka and occupancy to file evaluation.txt as KA_result, occupancy, 1, time_stamp
    with open(result_dir+'/../evaluation.txt', 'a+') as f:
        f.write(f'{KA_result},{occupancy},1,{time_stamp}\n')
    
if __name__ == '__main__':
    wall_sizes = [(1260,1365)]

    for i in range(2,3):
        generate_example5_best_pose_one_weight_one_size(0.2,wall_sizes[0],weight = 1,seed_number = i)
    