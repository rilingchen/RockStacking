import cv2
import numpy as np
import trimesh
from skimage.measure import regionprops

def is_valid_rotation(image, rotated_image):
    # rotation is valid if the size of the stone in both images is close
    return np.sum(image) * 0.9 < np.sum(rotated_image) < np.sum(image) * 1.1
def rotate_axis_align(image):
    """Rotate a stone image so that its longest axis is aligned with the x axis.
    Args:
        image (np.array): A stone image.
    Returns:
        np.array: The rotated stone image.
    """
    # compute the minimal bounding box
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the range [-90, 0)
    # as the angle is in the range [-90, 0), we need to correct the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # rotate the image by the angle around the center
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.WARP_FILL_OUTLIERS, borderMode=cv2.BORDER_REPLICATE)
    if not is_valid_rotation(image,rotated_image):
        return None
    return rotated_image, angle

def rotate(stone_ori,angle):
    stone_matrix = regionprops(stone_ori)[0].image
    stone_center_ori = regionprops(stone_ori)[0].centroid
    # add zero padding to the stone for rotation
    stone = np.pad(stone_matrix, abs(stone_matrix.shape[0]-stone_matrix.shape[1]), mode='constant').astype('uint8')
    # rotate the stone
    rot_mat = cv2.getRotationMatrix2D(
                [int(stone.shape[1]/2), int(stone.shape[0]/2)], angle, 1.0)
    img_rotated = cv2.warpAffine(
                stone, rot_mat, stone.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
    # move the stone to the corner through cropping
    cropped_matrix = regionprops(img_rotated)[0].image
    tranformed_matrix = cropped_matrix
    # the translation conducted
    stone_center_rot = regionprops(tranformed_matrix.astype('uint8'))[0].centroid
    translation = (stone_center_rot[0]-stone_center_ori[0],stone_center_rot[1]-stone_center_ori[1])

    # #check validity
    # if not is_valid_rotation(stone_ori,tranformed_matrix):
    #     return None
    return tranformed_matrix,translation

def rotate_312(stone_mesh,zxy_sequence = [0,0,0]):
    new_mesh = stone_mesh.copy()
    centroid = new_mesh.centroid
    to_origin = trimesh.transformations.translation_matrix([-centroid[0],-centroid[1],-centroid[2]])
    _ = new_mesh.apply_transform(to_origin)
    
    new_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2*zxy_sequence[0], [0,0,1]))
    new_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2*zxy_sequence[1], [1,0,0]))
    new_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2*zxy_sequence[2], [0,1,0]))
    
    bbox_min = new_mesh.bounds[0]
    to_positive = trimesh.transformations.translation_matrix([-bbox_min[0],-bbox_min[1],-bbox_min[2]])
    _ = new_mesh.apply_transform(to_positive)
    return new_mesh

def rotate_axis_aligh_3d(mesh):
    to_origin = mesh.bounding_box_oriented.primitive.transform
    to_origin = np.linalg.inv(to_origin)
    new_mesh = mesh.copy()
    _ = new_mesh.apply_transform(to_origin)
    
    # rotate such that the shortest axis is aligned with the z axis
    # if new_mesh.extents[0] < new_mesh.extents[1]:
    #     new_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0]))
    # if new_mesh.extents[0] < new_mesh.extents[2]:
    #     new_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0,0,1]))
    if new_mesh.extents[0] > new_mesh.extents[1]:
        new_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0,0,1]))
    if new_mesh.extents[0] > new_mesh.extents[2]:
        new_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0]))
    bbox_min = new_mesh.bounds[0]
    to_positive = trimesh.transformations.translation_matrix([-bbox_min[0],-bbox_min[1],-bbox_min[2]])
    _ = new_mesh.apply_transform(to_positive)
    return new_mesh

def move_to_positive_3d(mesh):
    new_mesh = mesh.copy()
    bbox_min = new_mesh.bounds[0]
    to_positive = trimesh.transformations.translation_matrix([-bbox_min[0],-bbox_min[1],-bbox_min[2]])
    _ = new_mesh.apply_transform(to_positive)
    return new_mesh

def rotate_min_max_dim(mesh,max_iteration = 1000):
    best_mesh = mesh.copy()
    best_max_dim = np.max(best_mesh.extents)
    for i in range(max_iteration):
        # rotate the mesh with a random rotation matrix
        random_rotation = trimesh.transformations.random_rotation_matrix()
        rotated_mesh = mesh.copy()
        _ = rotated_mesh.apply_transform(random_rotation)
        #get the maximun dimension of the rotated mesh in x,y,z
        max_dim = np.max(rotated_mesh.extents)
        if max_dim<best_max_dim:
            best_max_dim = max_dim
            best_mesh = rotated_mesh.copy()
    return best_mesh
