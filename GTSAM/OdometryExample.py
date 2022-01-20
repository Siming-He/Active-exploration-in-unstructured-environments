import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
import utils.savefactorgraph as sfg

def main():
    # create an empty nonlinear graph
    graph = gtsam.NonlinearFactorGraph()

    # create prior at origin
    priorMean = gtsam.Pose2(0.0, 0.0, 0.0)
    # create the noise for prior
    priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    # add the prior to our graph
    graph.add(gtsam.PriorFactorPose2(1, priorMean, priorNoise))
    
    # create odometry factors
    odometry = gtsam.Pose2(2.0, 0.0, 0.0)
    # create the noise for odometry
    odometryNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    # add odometry factors to our graph between consecutive poses
    graph.add(gtsam.BetweenFactorPose2(1, 2, odometry, odometryNoise))
    graph.add(gtsam.BetweenFactorPose2(2, 3, odometry, odometryNoise))

    print("\nFactor Graph:\n{}".format(graph))

    # set the wrong initial values for further estimation
    initial = gtsam.Values()
    initial.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
    initial.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
    initial.insert(3, gtsam.Pose2(4.1, 0.1, 0.1))
    print("\nInitial Estimate:\n{}".format(initial))

    # optimize using Levenberg-Marquardt optimization
    optimizerParams = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, optimizerParams)
    result = optimizer.optimize()
    print("\nFinal Result: \n{}".format(result))

    # calculate marginal covariance for all variables (pose)
    marginals = gtsam.Marginals(graph, result)
    for i in range (1, 4):
        print("X{} covariance:\n{}\n".format(i, marginals.marginalCovariance(i)))

    for i in range(1, 4):
        gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5, marginals.marginalCovariance(i))

    plt.axis('equal')

    # Save factor graph and estimation result
    sfg.save_factor_graph(graph, "OdometryExample")
    plt.savefig("OdometryExampleOutputs/OdometryExample.png")

    plt.show()

if __name__ == "__main__":
    main()