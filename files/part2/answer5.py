import os
import numpy as np
import open3d as o3d
import cv2
from matplotlib import pyplot as plt

def transform_points_to_camera(points, transformation_matrix):
  points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

  camera_points_homogeneous = np.dot(transformation_matrix, points_homogeneous.T).T

  return camera_points_homogeneous

def project_points_to_image(camera_points, intrinsic_matrix):

  projected_points_homogeneous = np.dot(intrinsic_matrix, camera_points.T).T

  projected_points = projected_points_homogeneous[:, :2] / projected_points_homogeneous[:, 2:]

  return projected_points

def project_pts_to_img_plane(points, transformation_matrix, intrinsic_matrix):

  camera_points = transform_points_to_camera(points, transformation_matrix)

  projected_points = project_points_to_image(camera_points, intrinsic_matrix)

  return projected_points


def build_camera_pose_matrix(camera_translation, camera_rotation):

  translation_vector = np.reshape(camera_translation, (3, 1))

  camera_pose = np.concatenate((camera_rotation, translation_vector), axis=1)
  camera_pose = np.vstack((camera_pose, np.array([0, 0, 0, 1])))

  return camera_pose


def compute_rotation_between_normals(reference_normal, target_normal):

  reference_normal = np.reshape(reference_normal, (3,))
  target_normal = np.reshape(target_normal, (3,))

  reference_normal = np.atleast_2d(reference_normal).T
  target_normal = np.atleast_2d(target_normal).T

  u, _, vh = np.linalg.svd(np.matmul(reference_normal, target_normal), full_matrices=False)

  return np.dot(u, vh)


def compute_camera_translation_from_normals(reference_normal, target_offset, reference_offset):

  reference_normal = np.reshape(reference_normal, (3,))
  target_offset = np.reshape(target_offset, (3,))
  reference_offset = np.reshape(reference_offset, (3,))

  reference_normal = np.atleast_2d(reference_normal).T

  projection_matrix = np.linalg.inv(np.dot(reference_normal, reference_normal.T))

  camera_translation = np.dot(projection_matrix, target_offset - reference_offset)

  return np.squeeze(camera_translation)


def build_camera_matrix(rotation_matrix, translation_vector):

  translation_vector = np.reshape(translation_vector, (3,))

  camera_matrix = np.concatenate((rotation_matrix, translation_vector.reshape(3, 1)), axis=1)
  camera_matrix = np.vstack((camera_matrix, [0, 0, 0, 1]))

  return camera_matrix

def camera(rotation_matrix, translation_vector):

  camera_matrix = build_camera_matrix(rotation_matrix, translation_vector)
  camera_origin = -np.dot(camera_matrix.T, np.array([0, 0, 0, 1]))[:3]

  return camera_origin


def load_point_cloud(filename, with_colors=False):

  pcd = o3d.io.read_point_cloud(filename)
  points = np.asarray(pcd.points)

  if with_colors:
    if hasattr(pcd, "colors"):
      colors = np.asarray(pcd.colors)
      return np.concatenate((points, colors), axis=1)
    else:
      print(f"Warning: Point cloud '{filename}' does not contain color information.")
      return points

  return points


def fit_plane_to_points(points): # compute plane parameters
  
  centroid = np.mean(points, axis=0)
  centered_points = points - centroid

  _, _, vh = np.linalg.svd(centered_points)
  normal = vh[-1]

  offset = -np.dot(normal, centroid)

  return normal, offset


def file_to_array(path):
    with open(path, 'r') as file:
        lines = file.readlines()[:-1]

    float_lines = [[float(x) for x in line.split()] for line in lines]

    return np.squeeze(float_lines)

normals_plane = [] # plane_normals
offsets = []
cam_normals = [] # camera normals
rot_mat = [] #rotation matrices
cam_trans = [] # camera translations
cam_rot = [] #camera rotations
proj_pts = [] #projected points


path_pcd = 'C:/sem 6/cv/assignment2/CV-A2-calibration/lidar_scans'
path_camera_parameters = 'C:/sem 6/cv/assignment2/CV-A2-calibration/camera_parameters'
path_images = 'C:/sem 6/cv/assignment2/CV-A2-calibration/camera_images'
intrinsic_matrix = file_to_array('CV-A2-calibration/camera_parameters/camera_intrinsic.txt')
files = []
for file in os.listdir(path_pcd):
    files.append(file.split('.')[0])


for i, filename in enumerate(files):
  # Load point cloud data
  points = load_point_cloud(os.path.join(path_pcd, filename) + '.pcd')
  plane_parameters = fit_plane_to_points(points)  # Combine normal and offset

  # Load camera parameters
  param_path = os.path.join(path_camera_parameters, filename) + '.txt'
  camera_normal, rotation_matrix, translation_vector = file_to_array(param_path)

  # Calculate camera pose
  camera_offset = np.dot(-rotation_matrix.T, translation_vector) @ camera_normal
  camera_translation = compute_camera_translation_from_normals(camera_normal, camera_offset, plane_parameters[1])
  camera_rotation = compute_rotation_between_normals(plane_parameters[0], camera_normal)
  transformation_matrix = build_camera_pose_matrix(camera_translation, camera_rotation)

  # Project points onto image plane
  projected_points = project_pts_to_img_plane(points, transformation_matrix, intrinsic_matrix)

  # Visualize results (altered logic)
  if i >= 25:
    continue

  original_image = cv2.imread(os.path.join(path_images, filename) + '.jpeg')
  annotated_image = original_image.copy()

  # Draw projected points directly on a copy (avoid modifying original image)
  for point in projected_points:
    int_point = tuple(map(int, point))
    cv2.circle(annotated_image, int_point, 3, (0, 255, 0), -1)

  # Improved layout for visualization (optional)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
  ax1.imshow(original_image)
  ax1.set_title('Original Image')
  ax1.axis('off')

  ax2.imshow(annotated_image)
  ax2.set_title('Annotated Image (Checkerboard Points)')
  ax2.axis('off')

  plt.suptitle(f"Processing Results for File: {filename}", fontsize=20)
  plt.tight_layout()
  plt.show()


cosine_distances = []
fig, axes = plt.subplots(1, 5, figsize=(15, 30), subplot_kw={'projection': '3d'})  # Pre-create subplots

for i, (idx, _) in enumerate(enumerate(np.random.choice(len(files), 5, replace=False))):
  lidar_normal = normals_plane[idx]
  camera_normal = cam_normals[idx]
  rotation_matrix = rot_mat[idx]
  transformed_lidar_normal = np.dot(rotation_matrix, lidar_normal)

  cosine_dist = np.dot(camera_normal, lidar_normal) / (np.linalg.norm(camera_normal) * np.linalg.norm(lidar_normal))
  cosine_distances.append(cosine_dist)

  # Access current axis using unpacking from enumerate
  ax = axes[i]
  ax.quiver(0, 0, 0, camera_normal[0], camera_normal[1], camera_normal[2], color='r', label='Camera Normal')
  ax.quiver(0, 0, 0, lidar_normal[0], lidar_normal[1], lidar_normal[2], color='b', label='LIDAR Normal')
  ax.quiver(0, 0, 0, transformed_lidar_normal[0], transformed_lidar_normal[1], transformed_lidar_normal[2], color='g', label='Transformed LIDAR Normal')
  ax.set_title(f"Image {i+1}")
  ax.set_xlim([-1, 1])
  ax.set_ylim([-1, 1])
  ax.set_zlim([-1, 1])
  ax.legend()

plt.tight_layout()  # Adjust spacing between subplots
plt.show()

plt.hist(cosine_distances, bins=20, color='yellow', alpha=0.7)
plt.title('Histogram Cos Dist Errors')
plt.xlabel('Cosine Distance')
plt.ylabel('Freq')
plt.grid(True)
plt.show()

print("Average Error:", np.mean(cosine_distances))
print("Standard Deviation:", np.std(cosine_distances))
