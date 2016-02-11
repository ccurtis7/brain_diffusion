Motivation: We are creating this python package to facilitate organization,
visualization, and manipulation of nanoparticle diffusion data in the brain.
This will promote more efficient future analyses with brain effective diffusion
(Deff) data to decipher trends, allowing us to perform more meaningful
experiments with the most mobile nanoparticles to research brain disease and
therapeutics.

1. merge_Deff_and_particle_property_data
    - combine our 2 data sets into 1 with all parameters included
2. calculate_Deff
    - calculate Deff at any time point, not just 1 second
    - create functionality to produce a new dataset with new Deff values
3. create_trajectory_data
    - generate dataset of trajectory data containing x and y location of any
    single nanoparticle as a function of time
4. plot_trajectories
    - create plots of x and y position of the nanoparticle through time
    - create interactive feature where you can see the entire trajectory, or
    the particle at a given time
5. plot_histogram
    - create a histogram plot of percent of particles in diffusion ranges
    - create overlay functionality for histograms of different particles for
    comparison purposes
    - create interactive features for the histogram so we can select bin size
6. standard_error
    - calculate the standard error of Deff for any time scales
