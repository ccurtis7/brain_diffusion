# Summary

Our current dataset was made available by our PI, Professor Nance.  The data is
a part of a larger dataset that was used in a student's Masters thesis, and as
such must remain confidential.  
We have four datasets that were are going to analyze.  The first three are
actually derivatives from one larger dataset that we unfortunately don't have
access to.
We can probably use subsets of the data for unit tests, but if not, we can
fabricate fake data.

* Dataset 1 is a table of particle properties.
* Dataset 2 is average mean square displacement data of particles as measured in
the rat brain.  These are average values of anywhere between 50-300 particles.
The full dataset that includes trajectories of all particles isn't accessible.
* Dataset 3 is effective diffusion coefficients for individual particles
measured at 1s.
* Dataset 4 is sample trajectories of the individual particles.


# Dataset 1: Particle Properties

20 particle formulations with 18 recorded properties. Includes:

* Particle type
* PEG/no PEG
* Surfactant
* Size
* Size standard error
* Zeta potential
* Zeta potential standard error
* PDI
* No. of brain slices used for each rat (Rats 1,2 and 3)
* No. of particles tracked in each rat (Rats 1,2 and 3)

# Dataset 2: Mean Square Displacement Data

Includes mean square displacement data over 10 seconds in 3 rats using all 20
particles formulations at 299 time points (20 * 3 + 1 columns by 299 rows)

# Dataset 3: Effective Diffusion Coefficient Data

Calculated effective diffusion coefficients for each individual particle that
was tracked in the 20 * 3 experiments.  As there were different numbers of
particles tracked in each experiment, there are a variable number of columns
(20 * 3 rows by variable columns).

# Dataset 4: Sample trajectories

23 sample trajectories of particles from all 20 * 3 experiments.  Includes the
following data:

* Time
* x coordinate
* y coordinate
* Frame No.
* Particle No.
