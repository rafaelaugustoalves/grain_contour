This is just a quick draft.

run "contour_paths.py" to do the pore conections.
run "renorm_tiff.py" to renormalize the tiff images (helps the pores to pop-out) and creat a hdf file with the segmentation.
run "visualize.py" to see segmentation. This is just to make sure the segmentation was saved right.

There are two envoriments because openCV (cv2) is not playing nicely with "dijkstra3d" (https://github.com/seung-lab/dijkstra3d).
"run_environment.yml" is the conda enviroment with dijkstra3d, which is needed to run the "contour_paths.py".
"tif_environment.yml" is the conda enviroment with openCV, which is needed for the other two scripts.
