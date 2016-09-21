# VGG16-Image-Retrieval
Uses TensorFlow and FC2 features to match test images to the same category given a query image as input

How to use:
* Install TensorFlow by following instructions here: https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html
* Uncompress the dataset folder
* Simply execute the vgg_example.py file
  * python vgg_example16.py

Once it is up and running, every so often you will see prints similar to the following:

Current Correct list: ['image_00748.jpg', 'image_00749.jpg', 'image_00750.jpg', 'image_00751.jpg']

Matches are:
 * Distance: 2.45867538452, FileName: image_00748.jpg
 * Distance: 22.8059749603, FileName: image_00751.jpg
 * Distance: 37.0147171021, FileName: image_00750.jpg
 * Distance: 38.9278831482, FileName: image_00749.jpg
 * Precision@4: 1.0

This shows the similarity distance between the query and the matches as well as the filename of the matched images. 
This also shows the precision@4 compared to the ground truth. A file containing these prints as well as final results
will be created as "Last_Run.txt"
