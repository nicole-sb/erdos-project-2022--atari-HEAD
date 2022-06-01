# Predict 4 direction action 
The Jupyternote follows the following steps:
1. For gaze positions, find start position, end position, use pca to find mean position, variances, and components(components are transformed to angles)
2. For each image, find the position of pac_man, and possible actions.
3. Calculate the angle and amplitude of (mean,start,end) gaze position relative to pac_man position in each frame.
4. Final features: ['variance0','variance1','com_angle0','com_angle1','mean_angle','mean_amplitude','start_angle','start_amplitude','end_angle','end_amplitude','0','1','2','3']
   Parameter of interest: 'action'
5. Use normal classification regression, multilayer neuro network to train the data. Now the best accuracy is 62% for test data. Maybe hyperparameters need to be adjusted.

--Jason
