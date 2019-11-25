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

The project uses numpy, matplotlib and networkx.

#### Numpy, scipy and matplotlib

`sudo apt-get install python-pip`
`python -m pip install --user numpy scipy matplotlib`

#### Networkx

`pip install networkx`

## Source code organization

 * src/: 
   * `Setup.py`: Contains the setup information of the robot teams and their organization graph
   * `TestIntermittent.py`: Main script to be executed, performs the intermittent communiction algorithm
   
   * Classes/:
     * `Robot.py`: Contains the robot class with all its attributes
     * `Schedule.py`: Creates the schedule and teams

   * Utilities/:
     * `ControllerUtilities.py`: Contains functions used in the main control loop update()
     * `PathPlanningUtilities.py`: Contains all the functions for the updatePath() function
     * `VisualizationUtilities.py`: Plotting functions for various data formats

   * Data/:
     * `FTLEDoubleGyre.jpg`: FTLE image used for the measurements
     * `FTLEDoubleGyre.mat`: FTLE matrix from matlab calculations
 
 * matlab/:
   * `double_gyre_func.m`: Velocity function of the double gyre
   * `double_gyre_generator.m`: Creates a double gyre
   * `double_gyre_trajectories.m`: Calculates the trajectories for the double gyre
   * `FTLE_computation.m`: Computes the FTLE of the double gyre, outputs the data required for the `Setup.py` file
   * `time_dep_double_gyre.m`: Time dependent double gyre
   

## Support

Please contact [Hannes Rovina](mailto:hannes1_rovina@hotmail.com) for technical support, or to report bugs.

## Contributing

Please clone the repository, make a new branch and have fun with the repo.

## Licence

This project is distributed under the `Apache License`, Version `2.0`. More information can be found in the `LICENSE` file.