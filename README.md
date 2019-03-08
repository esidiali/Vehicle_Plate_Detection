# Vehicle_Plate_Detection
[ENG]

In the license plate detection sequence, we propose three methods:


1- unsupervised method:

	-----> input arguments: only the path tp the image to be processed;

	-----> output: list of coordinates of the plate of the image (X, Y, Delta_X, Delta_Y).



2- supervised method based on random forests:
	
-----> input arguments: array  of the features of the images (after extraction of the HOG characteristics) + the corresponding annotations (ground truth);
	
-----> output: array of coordinates, same form as ground truth data: (X, Y, Delta_X, Delta_Y).



3- supervised method based on convolutional neural networks to draw characteristics.:
	-----> input arguments: like RF;
	
-----> output: as RF.



The performances of these methods are very different. The supervised methods are the best, since in the unsupervised method, 
we must define ourselves a set of criteria for extraction of the plate or its coordinates: fixed assumptions but variable images.



DATA: the dataset used : https://github.com/openalpr/benchmarks/tree/master/endtoend/
