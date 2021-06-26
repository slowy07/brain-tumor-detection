# brain tumor detection


## About the data
the datasets contain 2 folders : yes and no wich contains 253 brain MRI images. the folder yes contains 155 brain MRI images that are tumorous and the folder no contains 98 Brain MRI images that are non tumorous

## data preprocessing
For every image, the following preprocessing steps were applied:
1. crop the part of the image that contains only the brain (which is the most important part of the image)
2. resize the image to have shape of (240, 240, 3) = (image_width, image_height, number_of_channels) beause images in the datasets come in different size. so, all images should have the same shape to feed it as an input to the neural network
3. apply normalization to scale pixel value to the range 0-1

## data split
the data split in the following
 - 70% of the data for training
 - 15% of the data for validation
 - 15% of the data for testing