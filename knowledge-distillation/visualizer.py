import numpy as np
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
import torch

def student_inference(model, dataloader, voxel_size = 0.02, device="cuda"):
    pcds = []
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for image in dataloader:
            # Assuming the batch is a tuple of (inputs, targets)
            inputs = image.to(device)
            output = model(inputs)
            # c = output.shape[1]

            # Reshape and concatenate the point clouds from each frame
            point_cloud = output.view(-1, 3)
            print(point_cloud.shape)
            pointcloud = o3d.geometry.PointCloud \
                        (o3d.utility.Vector3dVector(point_cloud.cpu().numpy()))
            pointcloud = pointcloud.voxel_down_sample(voxel_size=voxel_size)

            pcds.append(pointcloud)

    return pcds


def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    target.estimate_normals()
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

def optimize_pose_graph(pcds, voxel_size):

    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    
    print("Full registration ...")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=max_correspondence_distance_fine,
                                                                edge_prune_threshold=0.25,
                                                                reference_node=0)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(pose_graph,
                                                        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                        option)
    return pose_graph


def visualize_point_cloud(pcds, voxel_size = 0.001):
    # Combine PCD and return
    pcd_combined = o3d.geometry.PointCloud()
    pose_graph = optimize_pose_graph(pcds, voxel_size)
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud("multiway_registration.ply", pcd_combined_down)


    return pcd_combined_down