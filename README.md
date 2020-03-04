# ReadMe

This repo contains the code for my master thesis about adaptive sampling with intermittent communication for heterogeneous robot teams.

## Environment Setup

First you need to install a couple libraries to be able to execute the code. The code is written in python version 3.7.1 under Ubuntu 18.04.

Setup instructions are executed using the command line so you will need to work with the terminal and you need sudo access. In order to generate the double gyre snapshot matlab is also required, but the data is already in the `src/Data` folder

### Project source code

First, clone the repository using the following command:  
`git@github.com:HannesRovina/AdaptiveSamplingIntermittentComms.git`

### Install python

`sudo apt-get update`

`sudo apt-get install python3.7`

### Install libraries

The project uses numpy, matplotlib, skimage, scipy, networkx and GPy.

#### Numpy, scipy, skimage and matplotlib

`sudo apt-get install python-pip`

`python -m pip install --user numpy scipy skimage matplotlib`

#### Networkx

`pip install networkx`


#### GPy

`pip install gpy`


## Source code organization

 * src/: 
   * `Setup.py`: Contains the setup information of the robot teams and their organization graph as well as starting locations and measurment data setup
   * `TestIntermittent.py`: Main script to be executed, performs the intermittent communiction algorithm
   
   * Classes/:
     * `Robot.py`: Contains the robot class with all its attributes
     * `Schedule.py`: Creates the schedule and teams
     * `GaussianProcess.py`: Creates the Gaussian Process for each robot
     * `SpatioTemporal.py`: Creates the custom kernel for the GP's

   * Utilities/:
     * `ControllerUtilities.py`: Contains functions used in the main control loop update()
     * `PathPlanningUtilities.py`: Contains all the functions for the updatePath() function
     * `VisualizationUtilities.py`: Plotting functions for various data formats
     * `LogUtilities.py`: Writes results to logfile
     * `ErrorAnalysis.py`: Script to be executed to analyses the errors 
     * `BariumCloudPreprocessing.py` Script to generate barium cloud data from image sequence

   * Data/:
     * `FTLEDoubleGyre.jpg`: FTLE image used for the measurements
     * `FTLEDoubleGyre.mat`: FTLE matrix from matlab calculations
     * meshfiles/:
       * `600x600_mesh.mat`: Defines the mesh locations
       * `600x600_node_soln_fine.mat`: Gives the measurements of solvant for each mesh node defined in `600x600_mesh.mat`
       * `600x600_node_soln_fine_times.mat`: Gives the measurement times of solvant for each mesh node defined in `600x600_mesh.mat`
     * BariumCloudImages/:
       * `barium_cloud_1_movie.mp4`: Barium cloud video to generate data
       * Processed/:
         * `BariumCloudDataSmall.npz`: Barium cloud data in original size
         * `BariumCloudDataBig.npz`: Barium cloud data in upsampled size
 
 * matlab/:
   * `double_gyre_func.m`: Velocity function of the double gyre
   * `double_gyre_generator.m`: Creates a double gyre
   * `double_gyre_trajectories.m`: Calculates the trajectories for the double gyre
   * `FTLE_computation.m`: Computes the FTLE of the double gyre, outputs the data required for the `Setup.py` file
   * `time_dep_double_gyre.m`: Time dependent double gyre
   
# Important Note:

The path variables need to be adjusted in order for the program to execute correctly!
Path variables in files:

`TestIntermittent.py`  
`Setup.py`  
`BariumCloudPreprocessing.py`  
`ErrorAnalysis.py`  

Also important, the file `SpatioTemporal.py` needs to be copied in the location of the kernels of the GPy package, in my case it was:  

`/home/hannes/anaconda3/lib/python3.7/site-packages/GPy/kern/src`  

Furthermore, in `BariumCloudPreprocessing.py` the terminal command described in the note has to be executed with the correct paths for video image sequencing.  

## Support

Please contact [Hannes Rovina](mailto:hannes1_rovina@hotmail.com) for technical support, or to report bugs.

## Contributing

Please clone the repository, make a new branch and have fun with the repo.

## Licence

This project is distributed under the `Apache License`, Version `2.0`. More information can be found in the `LICENSE` file.