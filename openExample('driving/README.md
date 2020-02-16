# Matlab Vehicle Detector and Localisation

CNN’s are primarily used to classify images, where as R-CNN’s, R standing for region, are used to detect objects. Traditional CNN’s can classify objects but cannot convey where they are. It is possible to regress bounding boxes using a CNN, but that is only possible for one object at a time. This is because regressing multiple boxes causes interference.
In R-CNN’s, one object is focused at a time, so a single object of interest dominates in a given region, this minimizes interference. The regions detected are then resized so that equally sized regions are fed into the CNN for classification and bounding box regression.

To train this model, a dataset consisting of 295 pre-labelled images of vehicles are used.
The images consist of one or more labelled instances of a vehicle.
The detector is trained in four steps. The first and second step train the region proposal
and detection networks. The third and fourth steps join the first two networks and creates
a single network for detection. The network training options are specified using training
options.

As the dataset consists of images with different sizes, the minibatch size is set to
This ensures that these images won’t be processed together. Setting a temporary location for
checkpoint path will allow you to pause training of the network and lets you later resume from where you left off. A pretrained CNN called ResNet-50 is used for feature extraction. On training the R-CNN with the specified parameters, the network was able to detect cars with an average accuracy of 86%. The
time taken to complete training totaled to 4 hours.

# Accuracy Rating

![image](https://user-images.githubusercontent.com/60957986/74599542-293a5e00-509d-11ea-9666-01d38b2e51ec.png)

