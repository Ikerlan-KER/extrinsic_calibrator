# ------------------------------------------------------------------------------
# This file is part of **extrinsic-calibrator:
# October 2024
# Copyright 2024 IKERLAN. All Rights Reserved.
#
#
# LICENSE NOTICE
#
# This software is available under a dual license system. Choose between:
# - GNU Affero General Public License v3.0 for open-source usage, or
# - A commercial license for proprietary development.
# For commercial license details, contact us at info@ikerlan.es.
#
# GNU Affero General Public License v3.0
# Version 3, 19 November 2007
# © 2007 Free Software Foundation, Inc. <https://fsf.org/>
#
# Licensed under a dual license system:
# 1. Open-source usage under the GNU Affero General Public License v3.0
#    (AGPL-3.0), allowing you to freely use, modify, and distribute the
#    software for open-source projects. You can find a copy of the AGPL-3.0
#    license at https://www.gnu.org/licenses/agpl-3.0.html.
# 2. For commercial/proprietary use, a separate commercial license is required.
#    Please contact us at info@ikerlan.es for inquiries about our commercial
#    licensing options.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Author Information:
# Author: Josep Rueda Collell
# Created: October 2024
# Affiliation: IKERLAN (https://www.ikerlan.es)
# ------------------------------------------------------------------------------

# Standard libraries
import copy
import random
from collections import deque

# Well-known libraries
import cv2
import numpy as np
from prettytable import PrettyTable

# ROS-specific imports
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
import tf_transformations

# Custom parameters
from extrinsic_calibrator_core.python_aruco_parameters import aruco_params
from extrinsic_calibrator_core.python_camera_topics_parameters import cameras_params



class ExtrinsicCalibrator(Node):
    def __init__(self):
        super().__init__('detector_aruco_node')
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # OpenCV bridge for converting ROS Image to OpenCV image
        self.bridge = CvBridge()
        
        aruco_params_listener = aruco_params.ParamListener(self)
        imported_aruco_params = aruco_params_listener.get_params()
        self.real_aruco_params = ArucoParams(self,imported_aruco_params)
        
        cameras_param_listener = cameras_params.ParamListener(self)
        self.imported_cameras_params = cameras_param_listener.get_params()
        
        # Get all cameras, filtering out those that start or end with '_'
        cam_attributes = [attr for attr in dir(self.imported_cameras_params) if not (attr.startswith('_') or attr.endswith('_'))]
        
        # construct the cameras
        self.array_of_cameras = []
        
        for camera_counter, attr_name in enumerate(cam_attributes):
            attr_value = getattr(self.imported_cameras_params, attr_name)
            
            # Check if the attribute has the 'image_topic' attribute before accessing it
            if not (hasattr(attr_value, 'image_topic') and hasattr(attr_value, 'image_topic')):
                self.get_logger().error(f"Skipping attribute '{attr_name}' due to missing 'image_topic' attribute.")
            else:
                self.array_of_cameras.append(Camera(self, attr_name, camera_counter, attr_value.image_topic, attr_value.camera_info_topic, self.bridge, self.tf_broadcaster, self.real_aruco_params))

        # periodically check if all cameras are calibrated        
        self.timer = self.create_timer(2.0, self.check_camera_transforms_callback)
        

    def check_camera_transforms_callback(self):
        if all([camera.are_all_transforms_precise() for camera in self.array_of_cameras]):
            self.get_logger().info(f"All marker transforms gathered successfully")
            for camera in self.array_of_cameras:
                camera:Camera
                self.get_logger().info(f"Camera {camera.camera_name} has received all marker transforms")
                camera.image_sub.destroy()
                camera.camera_info_sub.destroy()
            self.timer.cancel()
            self.initiate_calibration_routine()
            return True
        else:
            self.get_logger().warn(f"Not all marker transforms gathered successfully")
            return False


    def initiate_calibration_routine(self):
        # Routine to be executed command by command unless something goes wrong
        if not self.generate_is_marker_visible_from_camera_table():
            return False
        if not self.find_central_marker():
            return False
        if not self.generate_transform_between_markers_table():
            return False
        if not self.generate_path_between_markers_table():
            return False
        if not self.generate_reliable_transform_between_markers_table():
            return False
        if not self.broadcast_markers_transforms():
            return False
        if not self.generate_camera_to_marker_transform_table():
            return False
        if not self.generate_world_to_cameras_transform_table():
            return False
        if not self.broadcast_cameras_to_world():
            return False
        self.get_logger().info("Extrinsic calibration finished successfully.")
        self.get_logger().info("The transforms will remain alive while this Node remains too. Hit Ctrl+C to exit")
        
        # To keep the Transforms alive
        while(1):
            pass
        
        
    def generate_is_marker_visible_from_camera_table(self):  
        # Build a 2D array where we can see which markers are visible by each camera is_marker_visible_from_camera_table[camera_id][marker_id]
        self.largest_marker = max([max(camera.reliable_marker_transforms.keys()) for camera in self.array_of_cameras])
        self.is_marker_visible_from_camera_table = [[False for _ in range(len(self.array_of_cameras))] for _ in range(self.largest_marker + 1)]
        
        # Get array of all seen markers
        seen_markers = []
        for camera in self.array_of_cameras:
            for marker_id in camera.reliable_marker_transforms.keys():
                seen_markers.append(marker_id)
        
        # Fill is_marker_visible_from_camera_table
        for camera in self.array_of_cameras:
            for marker_id in seen_markers:
                if marker_id in camera.reliable_marker_transforms:
                    self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] = True
        
        # Display the is_marker_visible_from_camera_table
        self.display_camera_to_marker_table("Which markers does each camera see:", self.is_marker_visible_from_camera_table)
        
        # Check for exceptional cases
        for camera in self.array_of_cameras:
            # Check if a camera is not seeing any marker
            if all([not self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] for marker_id in range(self.largest_marker + 1)]):
                self.get_logger().error(f"Camera {self.array_of_cameras[camera.camera_id].camera_name} doesn't see any marker")
            # Check if a camera is only seeing one marker
            elif sum(self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] for marker_id in range(self.largest_marker + 1)) == 1:
                if (self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] for marker_id in range(self.largest_marker + 1)):
                    self.get_logger().warn(f"Camera {camera.camera_name} is only seeing one marker (Marker {marker_id})")
                    
        # Check if any row has all False values
        for marker_id in range(self.largest_marker + 1):
            # Check if a marker is not seen by any camera
            if all([not self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] for camera in self.array_of_cameras]):
                # self.get_logger().warn(f"Marker {marker_id} is not seen by any camera")
                pass
            # Check if a marker is olny seen by one camera
            elif sum(self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] for camera in self.array_of_cameras) == 1:
                if (self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] for camera in self.array_of_cameras):
                    self.get_logger().warn(f"Marker {marker_id} is only seen by one camera (Camera {camera.camera_name})")
                            
        # Check if specifically marker 0 is seen by any camera
        if not any(self.is_marker_visible_from_camera_table[0][camera.camera_id] for camera in self.array_of_cameras):
            self.get_logger().error(f"Marker 0 is not seen by any camera")
            return False
        
        else:
            return True
        
        
    def find_central_marker(self):
        # Table with the camera scores
        cameras_table = [[None for _ in range(len(self.array_of_cameras))] for _ in range(self.largest_marker + 1)]
        # Table with the marker scores
        markers_table = [[None for _ in range(len(self.array_of_cameras))] for _ in range(self.largest_marker + 1)]
        self.scores_table = [[None for _ in range(len(self.array_of_cameras))] for _ in range(self.largest_marker + 1)]
        
        # For each cell in the table, provide two tables, one indicating how many other markers are seen byt the same camera, and another indicating how many cameras sees one individual camera
        for camera in self.array_of_cameras:
            camera:Camera
            for marker_id in range(self.largest_marker + 1):
                camera_counter = 0
                marker_counter = 0
                for i in range(len(self.is_marker_visible_from_camera_table[marker_id])):
                    if self.is_marker_visible_from_camera_table[marker_id][i]:
                        camera_counter += 1
                for i in range(len(self.is_marker_visible_from_camera_table)):
                    if self.is_marker_visible_from_camera_table[i][camera.camera_id]:
                        marker_counter += 1
                if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]:
                    cameras_table[marker_id][camera.camera_id] = marker_counter-1
                    markers_table[marker_id][camera.camera_id] = camera_counter-1
                    
        # The total result will be the addition of the two tables
        for camera in self.array_of_cameras:
            camera:Camera
            for marker_id in range(self.largest_marker + 1):
                if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]:
                    self.scores_table[marker_id][camera.camera_id] = cameras_table[marker_id][camera.camera_id] + markers_table[marker_id][camera.camera_id]
                else:
                    self.scores_table[marker_id][camera.camera_id] = None
        
        # Display the scores_table
        self.display_camera_to_marker_table("Scores of each camera-marker couple:", self.scores_table)
        
        # Check which marker has better punctuation by checking the maximum value of the table and returning its indices
        self.center_marker = self.find_random_max_index(self.scores_table)
        self.get_logger().info(f"Our central marker is Marker {self.center_marker}")
        
        return True
        
        
    def find_random_max_index(self, scores_table):
        max_value = float('-inf')
        max_indices = []

        # Loop over the table to find the max value and its indices
        for marker_id in range(self.largest_marker + 1):
            for camera in self.array_of_cameras:
                camera:Camera
                if scores_table[marker_id][camera.camera_id] is not None:
                    value = scores_table[marker_id][camera.camera_id]
                    if value > max_value:
                        max_value = value
                        max_indices = [(marker_id, camera.camera_id)]  # Reset list if new max is found
                    elif value == max_value:
                        max_indices.append((marker_id, camera.camera_id))  # Add to list if value equals current max

        # Randomly select one of the max indices
        max_marker_id, max_camera_id = random.choice(max_indices)
        if scores_table[max_marker_id][max_camera_id] == 0:
            self.get_logger().error(f"The center marker has no links between cameras nor markers")
        return max_marker_id


    def generate_transform_between_markers_table(self):
        # Table with all the transforms we know between markers due to the camera transforms
        
        # Create one table connected markers by a single camera
        for camera in self.array_of_cameras:
            camera:Camera
            camera.can_camera_connect_two_markers_table = [[False for _ in range(self.largest_marker + 1)] for _ in range(self.largest_marker + 1)]
            for origin_marker_id in range(self.largest_marker + 1):
                for destination_marker_id in range(self.largest_marker + 1):
                    if origin_marker_id != destination_marker_id:
                        if self.is_marker_visible_from_camera_table[origin_marker_id][camera.camera_id] and self.is_marker_visible_from_camera_table[destination_marker_id][camera.camera_id]:
                            camera.can_camera_connect_two_markers_table[origin_marker_id][destination_marker_id] = True
        
        # Merge the previous table into a huge table to check which markers can be connected at all
        self.does_transform_exist_between_markers_table = [[False for _ in range(self.largest_marker + 1)] for _ in range(self.largest_marker + 1)]
        for camera in self.array_of_cameras:
            camera:Camera
            for origin_marker_id in range(self.largest_marker + 1):
                for destination_marker_id in range(self.largest_marker + 1):
                    if camera.can_camera_connect_two_markers_table[origin_marker_id][destination_marker_id] == True:
                        self.does_transform_exist_between_markers_table[origin_marker_id][destination_marker_id] = True
        
        # Print the does_transform_exist_between_markers_table
        self.display_marker_to_marker_table(f"Does exist a Transform between two markers through a camera:", self.does_transform_exist_between_markers_table)
            
        return True
   
        
    def generate_path_between_markers_table(self):
        # Table with all the different combination of paths to go from one marker to another
        self.path_between_markers_table = [[None for _ in range(self.largest_marker + 1)] for _ in range(self.largest_marker + 1)]
        for origin_marker_id in range(self.largest_marker + 1):
            for destination_marker_id in range(self.largest_marker + 1):
                self.path_between_markers_table[origin_marker_id][destination_marker_id] = self.explore_paths_between_markers(origin_marker_id, destination_marker_id)
        
        # Print the path_between_markers_table
        # self.display_marker_to_marker_table(f"Path between markers:", self.path_between_markers_table) 
        
        # Generate the even more expanded table of paths
        self.path_between_markers_with_cameras_table = [[ [] for _ in range(self.largest_marker + 1)] for _ in range(self.largest_marker + 1)]
        for origin_marker_id in range(self.largest_marker + 1):
            for destination_marker_id in range(self.largest_marker + 1):
                if self.path_between_markers_table[origin_marker_id][destination_marker_id] is not None:
                    for path in self.path_between_markers_table[origin_marker_id][destination_marker_id]:
                        new_paths = self.return_the_cameras_between_markers_in_path(path)
                        if new_paths is not None:
                            for path in new_paths:
                                self.path_between_markers_with_cameras_table[origin_marker_id][destination_marker_id].append(path)
                else:
                    self.path_between_markers_with_cameras_table[origin_marker_id][destination_marker_id] = None    

        # Print the path_between_markers_with_cameras_table
        # self.display_marker_to_marker_table(f"Path between markers including cameras:", self.path_between_markers_with_cameras_table)    
        
        return True
        # Check that all markers are connected with marker 0
        # TODO
    
    
    def explore_paths_between_markers(self, origin_marker_id, destination_marker_id):
        # Function to retrieve all the paths between markers
        if origin_marker_id == destination_marker_id:
            return None
        else:
            # Temporary starting path with the origin_marker_id
            array_of_paths = []
            current_path = []
            current_path.append(origin_marker_id)
            array_of_paths.append(current_path)        
            return self.evaluate_paths(origin_marker_id, destination_marker_id, array_of_paths)

            
    def evaluate_paths(self, origin_marker_id, destination_marker_id, current_array_of_paths):
        # Array of paths that successfully go from origin_marker_id to destination_marker_id
        array_of_successful_paths = []
        # Temporary array of paths that are still candidates to be added to the array_of_successful_paths
        array_of_paths = []
        array_of_paths = copy.deepcopy(current_array_of_paths)
        # For each path in the array of paths, find the children of the last element of the path. For each child, duplicate the path, adding the child at the end
        temp_array_of_paths = []
        for path in array_of_paths:
            children = self.return_children(path[-1])
            for child in children:
                new_path = copy.deepcopy(path)
                new_path.append(child)
                temp_array_of_paths.append(new_path)
        # Temporary array of paths that are still candidates to be added to the array_of_successful_paths, now with the children added       
        array_of_paths = copy.deepcopy(temp_array_of_paths)
        
        # Check if any of the paths can be added to the array of finished paths
        temp_array_of_paths = copy.deepcopy(array_of_paths)
        for path in temp_array_of_paths:
            # if the newly added child is the origin, pop the path out of the candidates array
            if path[-1] == origin_marker_id:
                array_of_paths.remove(path)
            # if the newly added child was already in the path, pop the path out the candidates array
            elif path[-1] in path[:-1]:
                array_of_paths.remove(path)
            # if the newly added child is the destination, add it to the array_of_successful_paths
            elif destination_marker_id == path[-1]:
                if path not in array_of_successful_paths:
                    array_of_successful_paths.append(path)
                array_of_paths.remove(path)
        
        # If there are still candidates in the array_of_paths iterate recursively on those
        if len(array_of_paths) > 0:
            new_good_paths = self.evaluate_paths(origin_marker_id, destination_marker_id, array_of_paths)
            if new_good_paths is not None:
                for new_path in new_good_paths:
                    if new_path not in array_of_successful_paths:
                        array_of_successful_paths.append(new_path)
        
        # If no path was found return None                
        if array_of_successful_paths is None:
            return None
        elif len(array_of_successful_paths) == 0:
            return None
        else:
            return array_of_successful_paths
            
       
    def return_children(self, origin_marker):
        # Function to simply obtain the children of a marker, meaning the markers that can be reached from the origin_marker
        can_transform = []
        for child_marker in range(self.largest_marker + 1):
            if self.does_transform_exist_between_markers_table[origin_marker][child_marker]:
                can_transform.append(child_marker)
        return can_transform
    
    
    def return_the_cameras_between_markers_in_path(self,path):
        # Function to insert the cameras in the paths, needed to transform from anoe to another
        final_array_of_paths = []
        
        paths_to_evaluate = []
        temp_path = copy.deepcopy(path)
        paths_to_evaluate.append(temp_path)
        
        for path_being_evaluated in paths_to_evaluate:
            # Check if the path has been evaluated by checking if the last element is a number, and the previous a string
            if type(path_being_evaluated[-2]) is str and type(path_being_evaluated[-1]) is int:
                # If it is, then the path has been evaluated
                if path_being_evaluated not in final_array_of_paths:
                    final_array_of_paths.append(path_being_evaluated)
                paths_to_evaluate.pop(paths_to_evaluate.index(path_being_evaluated))
            else:
                # If it is not, then the path has not been evaluated
                # Find the first element in the path which next element is not a string
                for marker_id_index in range(len(path_being_evaluated)-1):
                    if type(path_being_evaluated[marker_id_index]) is int and type(path_being_evaluated[marker_id_index + 1]) is not str:
                        for camera in self.array_of_cameras:
                            camera:Camera
                            if self.is_marker_visible_from_camera_table[path_being_evaluated[marker_id_index]][camera.camera_id] and self.is_marker_visible_from_camera_table[path_being_evaluated[marker_id_index + 1]][camera.camera_id]:
                                new_path = copy.deepcopy(path_being_evaluated)
                                new_path.insert(marker_id_index + 1, camera.camera_name)
                                paths_to_evaluate.append(new_path)
                        
                        break     
                    else:
                        continue                      
                  
        if len(final_array_of_paths) > 0:
            return final_array_of_paths
        else:
            return None


    def generate_reliable_transform_between_markers_table(self):
        # Table with all the reliable transforms between markers. It tries to use as many transforms as possible in order to reduce the error
        self.reliable_transform_between_markers_table = [[None for _ in range(self.largest_marker + 1)] for _ in range(self.largest_marker + 1)]
        for origin_marker_id in range(self.largest_marker + 1):
            for destination_marker_id in range(self.largest_marker + 1):
                if self.path_between_markers_with_cameras_table[origin_marker_id][destination_marker_id] == None:
                    self.reliable_transform_between_markers_table[origin_marker_id][destination_marker_id] = None
                else:
                    self.reliable_transform_between_markers_table[origin_marker_id][destination_marker_id] = self.compose_marker_to_marker_transform(self.path_between_markers_with_cameras_table[origin_marker_id][destination_marker_id])
        
        # Print the reliable_transform_between_markers_table
        # self.display_marker_to_marker_table(f"Reliable transform between markers:", self.reliable_transform_between_markers_table) 
        
        return True
        
        
    def compose_marker_to_marker_transform(self, paths):
        # Function to provide us with the reliable transform between two markers, trying to use as many different paths as possible in order to reduce the error
        all_paths_transforms = []
        for path in paths:
            # reverse de path
            # We are, instead, getting the reverse path so we can sue the super magic powers of the pseudoinverse later
            path.reverse()
            path_transform = np.eye(4)
            for element_index in range(len(path) - 2):
                # the elements are intercalated, between cameras and markers so we will iterate accordingly
                # check if the element_index is even (meaning it corresponds to a marker)
                if element_index % 2 == 0:
                    origin_marker_id = path[element_index]
                    camera_name = path[element_index+1]
                    destination_marker_id = path[element_index + 2]
                    for camera in self.array_of_cameras:
                        if camera.camera_name == camera_name:
                            camera:Camera
                            if camera.can_camera_connect_two_markers_table[origin_marker_id][destination_marker_id]:
                                transform = np.dot(np.linalg.inv(camera.reliable_marker_transforms[origin_marker_id]), camera.reliable_marker_transforms[destination_marker_id])
                                path_transform = np.dot(path_transform, transform)
                            else:
                                self.get_logger().error(f"The table camera.can_camera_connect_two_markers_table of the camera {camera.camera_name} has a mistake")
                else:
                    continue
            all_paths_transforms.append(path_transform)
        stacked_transform = np.hstack(all_paths_transforms)
            
        # Calculate the number of 4x4 blocks in stacked_transform
        num_blocks = stacked_transform.shape[1] // 4
        stacked_identity = np.hstack([np.eye(4) for _ in range(num_blocks)])
        
        # Compute pseudoinverse and perform final multiplication
        pseudoinverse_result = np.linalg.pinv(stacked_transform)
        reliable_marker_to_marker_transform = np.dot(stacked_identity, pseudoinverse_result)
        return reliable_marker_to_marker_transform

                 
    def broadcast_markers_transforms(self):
        # Broadcast the transform between "marker_0" and "map"
        origin_transform = np.eye(4)
        self.broadcast_single_transform(f"marker_0", f"map", origin_transform)
        
        # Then broadcast all the transforms between the center_marker and the rest of the markers
        for destination_marker_id in range(self.largest_marker + 1):
            if self.reliable_transform_between_markers_table[self.center_marker][destination_marker_id] is not None:
                self.broadcast_single_transform(f"marker_{self.center_marker}", f"marker_{destination_marker_id}", self.reliable_transform_between_markers_table[self.center_marker][destination_marker_id])
        return True
    
        
    def broadcast_single_transform(self, origin_marker, destination_marker, transform):
        # Create a TransformStamped message
        t = TransformStamped()
            
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = origin_marker
        t.child_frame_id = destination_marker
        
        translation = tf_transformations.translation_from_matrix(transform)
        quaternion = tf_transformations.quaternion_from_matrix(transform)
        
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]
    
        self.tf_broadcaster.sendTransform(t)


    def generate_camera_to_marker_transform_table(self):
        # Create a table with the transforms between the cameras and the markers
        self.camera_to_marker_transform_table = [[None for _ in range(len(self.array_of_cameras))] for _ in range(self.largest_marker + 1)]
        for camera in self.array_of_cameras:
            camera:Camera
            for marker_id in range(self.largest_marker + 1):
                if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]:
                    self.camera_to_marker_transform_table[marker_id][camera.camera_id] = camera.reliable_marker_transforms[marker_id]
                else:
                    self.camera_to_marker_transform_table[marker_id][camera.camera_id] = None             
                   
        return True
        
        
    def generate_world_to_cameras_transform_table(self):
        # Create a table with the transforms between "map" and the cameras
        self.map_to_cameras_transform_table = [None for _ in range(len(self.array_of_cameras))]
        for camera in self.array_of_cameras:
            camera:Camera
            # check if the self.does_transform_exist_between_markers_table[camera_id]: has any True marker
            for marker_id in range(self.largest_marker + 1):
                if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]:
                    self.map_to_cameras_transform_table[camera.camera_id] = self.compose_world_to_camera_transform(camera.camera_id)
                    break
                else:
                    self.map_to_cameras_transform_table[camera.camera_id] = None
        
        return True


    def compose_world_to_camera_transform(self, camera_id):
        # Computes the transform between "map" and the camera trying to use as many camera transforms as possible
        array_of_camera_to_world_transforms = []
        for marker_id in range(self.largest_marker + 1):
            if self.is_marker_visible_from_camera_table[marker_id][camera_id]:
                if marker_id == 0:
                    marker_to_world_transform = np.eye(4)
                elif self.does_transform_exist_between_markers_table[marker_id][0]:
                    marker_to_world_transform = self.reliable_transform_between_markers_table[marker_id][0]
                else:
                    continue
                camera_to_marker_transform = self.camera_to_marker_transform_table[marker_id][camera_id]
                camera_to_world_transform = np.dot(camera_to_marker_transform,marker_to_world_transform)

                array_of_camera_to_world_transforms.append(camera_to_world_transform)
                
        stacked_transform = np.hstack(array_of_camera_to_world_transforms)
        # Calculate the number of 4x4 blocks in stacked_transform
        num_blocks = stacked_transform.shape[1] // 4
        stacked_identity = np.hstack([np.eye(4) for _ in range(num_blocks)])
        
        # Compute pseudoinverse and perform final multiplication
        pseudoinverse_result = np.linalg.pinv(stacked_transform)
        reliable_camera_transform = np.dot(stacked_identity, pseudoinverse_result)       

        return reliable_camera_transform


    def broadcast_cameras_to_world(self):
        transforms = []
        for camera in self.array_of_cameras:
            camera:Camera
            if self.map_to_cameras_transform_table[camera.camera_id] is not None:
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "map"
                t.child_frame_id = camera.camera_name
                
                transform = self.map_to_cameras_transform_table[camera.camera_id]
                translation = tf_transformations.translation_from_matrix(transform)
                quaternion = tf_transformations.quaternion_from_matrix(transform)
                
                t.transform.translation.x = translation[0]
                t.transform.translation.y = translation[1]
                t.transform.translation.z = translation[2]
                t.transform.rotation.x = quaternion[0]
                t.transform.rotation.y = quaternion[1]
                t.transform.rotation.z = quaternion[2]
                t.transform.rotation.w = quaternion[3]
                
                transforms.append(t)
        
        # Broadcast all transforms at once
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)
        return True


    def display_camera_to_marker_table(self, title, table_data):
        table = PrettyTable()
        table.field_names = ["Marker ID"] + [f"Camera {i}" for i in range(len(self.array_of_cameras))]
        if type(table_data[0][0]) is bool:
            for marker_id, row in enumerate(table_data):
                table.add_row([marker_id] + ['✓' if cell else '✗' for cell in row])
        else:
            for marker_id, row in enumerate(table_data):
                table.add_row([marker_id] + [cell for cell in row])
        self.get_logger().info(f"{title}\n" + table.get_string())


    def display_marker_to_marker_table(self, title, table_data):
        table = PrettyTable()
        table.field_names = ["Marker ID"] + [f"Marker {i}" for i in range(self.largest_marker + 1)]
        if type(table_data[0][0]) is bool:
            for marker_id, row in enumerate(table_data):
                table.add_row([marker_id] + ['✓' if cell else '✗' for cell in row])
        else:
            for marker_id, row in enumerate(table_data):
                table.add_row([marker_id] + [cell for cell in row])
        self.get_logger().info(f"{title}\n" + table.get_string())



class ArucoParams():
    def __init__(self, node:Node, aruco_params):
        if hasattr(cv2.aruco, aruco_params.aruco_dict):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_params.aruco_dict))
        else:
            node.get_logger().error(f"cv2.aruco doesn't have a dictionary with the name '{aruco_params.aruco_dict}'")
        self.marker_length = aruco_params.marker_length

        
            
class Camera():
    def __init__(self, node:Node, camera_name:str, camera_id:int, image_topic:str, camera_info_topic:str, bridge:CvBridge, broadcaster:tf2_ros.TransformBroadcaster, aruco_params:ArucoParams):
        
        self.node = node
        self.camera_name = camera_name
        self.camera_id = camera_id
        self.image_topic = image_topic
        self.camera_info_topic = camera_info_topic
        self.bridge = bridge
        self.tf_broadcaster = broadcaster
               
        self.node.get_logger().info(f"Camera {self.camera_name} created.")
               
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Define Aruco marker properties
        self.aruco_dict = aruco_params.aruco_dict
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.marker_length = aruco_params.marker_length  # length of the marker side in meters (adjust as needed)

        # Subscribe to the camera image topic and camera info
        self.image_sub = self.node.create_subscription(Image, image_topic, self.image_callback, 1)
        self.camera_info_sub = self.node.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 1)
        self.cv2_image_publisher = self.node.create_publisher(Image, f"{image_topic}/detected_markers", 10)
        
        self.marker_transforms = {}
        self.reliable_marker_transforms = {}


    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.node.get_logger().info(f"Camera {self.camera_name} parameters received.")


    def image_callback(self, msg):       
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.node.get_logger().warn(f"Camera {self.camera_name} parameters not yet received.")
            return
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # For ArUco detection, you can use the filtered_image directly
        corners, ids, rejected_img_points = self.detector.detectMarkers(cv_image)
        detected_ids = set()
        if ids is not None:
            for i, id in enumerate(ids):
                marker_id = id[0]
                detected_ids.add(marker_id)
                
                if marker_id not in self.marker_transforms and marker_id not in self.reliable_marker_transforms:
                    self.marker_transforms[marker_id] = deque(maxlen=30)

                objPoints = np.array([  [-self.marker_length/2, self.marker_length/2, 0],
                                        [self.marker_length/2,  self.marker_length/2, 0],
                                        [self.marker_length/2, -self.marker_length/2, 0],
                                        [-self.marker_length/2,-self.marker_length/2, 0]], dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(objPoints, corners[i], self.camera_matrix, self.dist_coeffs)
                if success:
                    rot_matrix, _ = cv2.Rodrigues(rvec)
                    translation_matrix = np.eye(4)
                    translation_matrix[:3, :3] = rot_matrix
                    translation_matrix[:3, 3] = tvec.flatten()
                    
                    # Draw the transform
                    cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length/2)
                    ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                    self.cv2_image_publisher.publish(ros_image)
                    
                    # Filter out the already reliable markers
                    if marker_id in self.reliable_marker_transforms:
                        continue
                    else:
                        self.marker_transforms[marker_id].append(translation_matrix)

                # Add None for markers not detected in this frame
                for marker_id in self.marker_transforms:
                    if marker_id not in detected_ids:
                        # Restart the precision of the marker if not seen
                        # self.marker_transforms[marker_id].append(None)
                        pass
                    
                # iterate through each marker of the marker_transforms dictionary
                for marker_id, transforms in self.marker_transforms.items():
                    if len(transforms) == 30:
                        self.check_precision(marker_id, transforms)
                
                # delete all the transforms from the marker_transforms dictionary    
                for marker_id, transform in self.reliable_marker_transforms.items():
                    if marker_id in self.marker_transforms:
                        del self.marker_transforms[marker_id]


    def check_precision(self, marker_id, transform):
        if self.is_precise(transform):
            self.node.get_logger().info(f"Camera {self.camera_name}: Marker {marker_id} is reliable")
            # add the last transform of the array in the dictionary as reliable marker transform
            self.reliable_marker_transforms[marker_id] = transform[-1]


    def is_precise(self, transforms):
        if all(transform is not None for transform in transforms):
            positions = np.array([t[:3, 3] for t in transforms])
            rotations = np.array([tf_transformations.euler_from_matrix(t) for t in transforms])

            position_range = np.ptp(positions, axis=0)
            rotation_range = np.ptp(rotations, axis=0)

            return np.all(position_range < 0.02) and np.all(rotation_range < np.radians(10))
        else:
            return False
       
        
    def are_all_transforms_precise(self):
        if len(self.reliable_marker_transforms) > 0 and len(self.marker_transforms) == 0:
            self.node.get_logger().info(f"Camera {self.camera_name}: All markers are reliable")
            return True
        else:
            for marker_id, transform in self.marker_transforms.items():
                if marker_id not in self.reliable_marker_transforms.keys():
                    self.node.get_logger().warn(f"Camera {self.camera_name}: Marker {marker_id} is not reliable, yet")
            return False
        
            
            