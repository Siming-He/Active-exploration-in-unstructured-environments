"""
Section 4 of GTSAM Tech Report: PoseSLAM
https://gtsam.org/tutorials/intro.html#magicparlabel-65467
"""

from __future__ import print_function
import math
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import utils.savefactorgraph as sfg

def main():
    # 1. create an empty nonlinear graph
    graph = gtsam.NonlinearFactorGraph()

    # 2a. add a prior on the first pose, setting it to the origin
    # create the prior noise model
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(0.3, 0.3, 0.1))
    
    graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), prior_noise))

    # 2b. add odometry factors
    # create the odometry noise model
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(0.2, 0.2, 0.1))

    graph.add(
        gtsam.BetweenFactorPose2(1, 2, gtsam.Pose2(2, 0, 0), odometry_noise))
    graph.add(
        gtsam.BetweenFactorPose2(2, 3, gtsam.Pose2(2, 0, math.pi / 2),
                                 odometry_noise))
    graph.add(
        gtsam.BetweenFactorPose2(3, 4, gtsam.Pose2(2, 0, math.pi / 2),
                                 odometry_noise))
    graph.add(
        gtsam.BetweenFactorPose2(4, 5, gtsam.Pose2(2, 0, math.pi / 2),
                                 odometry_noise))

    # 2c. add the loop closure constraint
    graph.add(
        gtsam.BetweenFactorPose2(5, 2, gtsam.Pose2(2, 0, math.pi / 2),
                                 odometry_noise))
    
    print("\nFactor Graph:\n{}".format(graph))

    # 3. set initial values of positions
    initial_estimate = gtsam.Values()
    initial_estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
    initial_estimate.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
    initial_estimate.insert(3, gtsam.Pose2(4.1, 0.1, math.pi / 2))
    initial_estimate.insert(4, gtsam.Pose2(4.0, 2.0, math.pi))
    initial_estimate.insert(5, gtsam.Pose2(2.1, 2.1, -math.pi / 2))
    
    print("\nInitial Estimate:\n{}".format(initial_estimate)) 

    # 4. optimize the initial calues using Gauss-Newton nonlinear optimizer 
    # or Levenberg-Marquardt optimizer
    # The LMA interpolates between the Gauss–Newton algorithm (GNA) 
    # and the method of gradient descent. 
    # The LMA is more robust than the GNA, which means that in many cases 
    # it finds a solution even if it starts very far off the final minimum. 
    # For well-behaved functions and reasonable starting parameters, 
    # the LMA tends to be slower than the GNA. 
    parameters = gtsam.GaussNewtonParams()
    parameters.setRelativeErrorTol(1e-5)
    parameters.setMaxIterations(100)
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, parameters)

    result = optimizer.optimize()
    print("Final Result:\n()".format(result))

    # 5. calculate and print marginal covariances for all variables
    marginals = gtsam.Marginals(graph, result)
    for i in range(1, 6):
        print("X{} covariance:\n{}\n".format(i, marginals.marginalCovariance(i)))
    
    for i in range(1, 6):
        gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5,
                              marginals.marginalCovariance(i))

    plt.axis('equal')

    # 6. Save factor graph and estimation result
    sfg.save_factor_graph(graph, "Pose2SLAM")
    plt.savefig("Pose2SLAMOutputs/Pose2SLAM.png")

    plt.show()

if __name__ == "__main__":
    main()