# results here should match sba_demo.py
# using camrig with 2 recitified sensors should match results from scam

import numpy as np
import g2o 

from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--noise', dest='pixel_noise', type=float, default=1.,
    help='noise in image pixel space (default: 1.0)')
parser.add_argument('--outlier', dest='outlier_ratio', type=float, default=0.,
    help='probability of spuroius observation  (default: 0.0)')
parser.add_argument('--robust', dest='robust_kernel', action='store_true', help='use robust kernel')
parser.add_argument('--dense', action='store_true', help='use dense solver')
parser.add_argument('--seed', type=int, help='random seed', default=0)
parser.add_argument('--n-sensors', type=int, help='number of sensors (currenlty at most 3)', default=2)
args = parser.parse_args()



def main():    
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)


    true_points = np.hstack([
        np.random.random((500, 1)) * 3 - 1.5,
        np.random.random((500, 1)) - 0.5,
        np.random.random((500, 1)) + 3])


    focal_length = (500, 500)
    principal_point = (320, 240)
    baseline = 0.075
    for frame_id in range(args.n_sensors):
        g2o.VertexCamRig.set_cam(frame_id, *focal_length, *principal_point)
        sensor_pose = g2o.Isometry3d(np.identity(3), [frame_id*baseline, 0, 0])
        g2o.VertexCamRig.set_calibration(frame_id, sensor_pose)

    true_poses = []
    num_pose = 5
    for i in range(num_pose):
        # pose here transform points from world coordinates to camera coordinates
        pose = g2o.Isometry3d(np.identity(3), [i*0.04-1, 0, 0])
        true_poses.append(pose)

        v_se3 = g2o.VertexCamRig()
        v_se3.set_id(i)
        v_se3.set_estimate(pose)
        if i < 2:
            v_se3.set_fixed(True)
        optimizer.add_vertex(v_se3)


    point_id = num_pose
    inliers = dict()
    sse = defaultdict(float)

    for i, point in enumerate(true_points):
        visible = []
        for j in range(num_pose):
            sens_vis = 0
            projections = []
            for frame_id in range(args.n_sensors):
                z = optimizer.vertex(j).map_point(frame_id, point)
                if 0 <= z[0] < 640 and 0 <= z[1] < 480:
                    projections.append((frame_id, z))
            if len(projections)>1:
                visible.append((j, projections))

        if len(visible) < 2:
            continue

        vp = g2o.VertexSBAPointXYZ()
        vp.set_id(point_id)
        vp.set_marginalized(True)
        vp.set_estimate(point + np.random.randn(3))
        optimizer.add_vertex(vp)

        inlier = True
        if np.random.random() < args.outlier_ratio:
            inlier = False
        if inlier:
            inliers[point_id] = i
            error = vp.estimate() - true_points[i]
            sse[0] += np.sum(error**2)
        for j, projections  in visible:
            for frame_id, z in projections:
                if not inliers:
                    z = np.array([
                        np.random.uniform(64, 640),
                        np.random.uniform(0, 480)])
                z += np.random.randn(2) * args.pixel_noise * [1, 1]

                edge = g2o.Edge_XYZ_VRIG()
                edge.set_vertex(0, vp)
                edge.set_vertex(1, optimizer.vertex(j))
                edge.set_measurement(np.hstack((z,[frame_id])))
                edge.set_information(np.identity(3))
                if args.robust_kernel:
                    edge.set_robust_kernel(g2o.RobustKernelHuber())

                edge.set_parameter_id(0, 0)
                optimizer.add_edge(edge)

        point_id += 1

    print ('num points', len(inliers))

    print('Performing full BA:')
    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(10)


    for i in inliers:
        vp = optimizer.vertex(i)
        error = vp.estimate() - true_points[inliers[i]]
        sse[1] += np.sum(error**2)

    print('\nRMSE (inliers only):')
    print('before optimization:', np.sqrt(sse[0] / len(inliers)))
    print('after  optimization:', np.sqrt(sse[1] / len(inliers)))
                    
    sse = defaultdict(float)
    for i,gt_pose in enumerate(true_poses):
        vp = optimizer.vertex(i)
        error = vp.estimate().translation() - pose.translation()
        sse[1] += np.sum(error**2)

    print('\nRMSE (inliers only):')
    print('pose error before optimization:', np.sqrt(sse[0] / len(inliers)))
    print('pose error after  optimization:', np.sqrt(sse[1] / len(inliers)))


if __name__ == '__main__':
    if args.seed > 0:
        np.random.seed(args.seed)

    main()
