# Data handling
The Jupyternote follows the following steps:
1. For gaze positions, find start position, end position, use pca to find mean position, variances, and components(components are transformed to angles)
2. Calculate the angle and amplitude of (mean,start,end) gaze position relative to pac_man position in each frame.
3. In each image, find the position of pac_man, nex possible actions. And find the nearest ghost position.
5. Final features: ['gaze_variance0','gaze_variance1',',ghost_amp','ghost_angle','mean_angle','mean_amplitude','start_angle','start_amplitude','end_angle','end_amplitude','com_angle0','com_angle1','0','1','2','3']
   Parameter of interest: 'action'
--Jason
