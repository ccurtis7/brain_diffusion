# brain-diffusion
Simple program that analyzes mean squared displacement data to develop optimal diffusion (in brain) parameters of nanoparticles

## User Instructions:

For the purposes of easy class grading, the final, polished functions for this analysis are located in the folders Ian_functions_and_unittests (for the timepoint-dependent Deff analysis), Rick_functions_and_unittests (for the interactive particle subchemistry histogram), and Chad_functions_and_unittests (for the particle trajectory plotting and animations).  For each of these, the data to be analyzed should be located in the working directory: that is, each of the aforementioned folders.  The data are not published at this time, and are therefore not posted to this public repository.  Please receive the data by email and copy them to the three folders.  Following this, these functions should be called to best demonstrate the analysis this code performs on our data:

##### Timepoint-dependent Deffs:
time_variable_Deff.compute_plot_all_Deff(tmin, tmax, particle_chemistry)

##### Interactive particle subchemistry histogram:
functions_rick.interact_plot_deff()

##### Particle trajectory plotting and animations:
trajectory_visualization.plot_trajectory(xydata, charttitle)
trajectory_visualization.sidebyside(xydata1, xydata2, charttitle1, charttitle2)
trajectory_visualization.overlay(xydata1, xydata2, charttitle)
animated_plot.py (run through bokeh server)
animated_overlay.py (run through bokeh server)
animated_sidebyside.py (run through bokeh server)

Within each folder, we have created demonstration IPython notebooks for ease of use--feel free to make use of these to explore our data and functionality.  Additional instructions can be found in these notebooks and in the docstrings of the individual files and functions.

**License note:** our license selection was rather careful, since the datasets we will be working with are unpublished and would ideally only be released to the public after peer-review and publication.  This would seem to suggest selection of a more restrictive license, but we think we have devised a way to keep the datasets themselves stored locally (with program operations calling those local data when needed), while pushing the program code to the repository to maintain its availability for open-source work.  In light of this setup, we selected a Simplified BSD License because of its ease of integration into new project contexts in the future.  This will allow maximal usability in a collaborative academic sense, enabling more widespread improvement of the code and of the science in this field.  More importantly, we would like to retain the option of using elements of this code in the activities of any future for-profit startup company based upon this science.  Such a startup, if it even happens, would occur years down the road, and tracking down the permissions to change to a BSD-style license at that point would be a nuisance (if we started with a GPL-style license).  So in our license selection we aim to maximize collaborative usability of the code, while retaining privately for now the data on which it works.
