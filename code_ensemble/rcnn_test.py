import keras_rcnn
from keras_rcnn import backend, datasets, layers, models, preprocessing
import numpy
import matplotlib
import keras

img_size=256
batch_size=1

training, test = datasets.load_data('DSB2018')

classes = {
    "nucleus": 1
}

# print(type(training))
# print(training[0])

for item in training:
	item['shape'] = (item['image']['shape']['r'], item['image']['shape']['c'], item['image']['shape']['channels'])
	item['filename'] = item['image']['pathname']
	item['boxes'] = item['objects']
	for x in item['boxes']:
		x['x1'] = x['bounding_box']['minimum']['c']
		x['x2'] = x['bounding_box']['maximum']['c']
		x['y1'] = x['bounding_box']['minimum']['r']
		x['y2'] = x['bounding_box']['maximum']['r']
generator = preprocessing.ObjectDetectionGenerator()

generator = generator.flow(training, classes, batch_size=batch_size) # target_shape=(img_size, img_size), scale=1, batch_size=batch_size)

for x in range(0, 5):
	(target_bounding_boxes, target_image, target_scores, _), _ = generator.next()

	target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

	target_image = numpy.squeeze(target_image)

	target_scores = numpy.argmax(target_scores, -1)

	target_scores = numpy.squeeze(target_scores)

	_, axis = matplotlib.pyplot.subplots(1, figsize=(12, 8))

	axis.imshow(target_image)

	for target_index, target_score in enumerate(target_scores):
	    if target_score > 0:
	        xy = [
	            target_bounding_boxes[target_index][0],
	            target_bounding_boxes[target_index][1]
	        ]

	        w = target_bounding_boxes[target_index][2] - target_bounding_boxes[target_index][0]
	        h = target_bounding_boxes[target_index][3] - target_bounding_boxes[target_index][1]

	        rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")

	        axis.add_patch(rectangle)

	matplotlib.pyplot.show()

image = keras.layers.Input((None, None, 3))

model = keras_rcnn.models.RCNN(image, classes=len(classes) + 1)

optimizer = keras.optimizers.Adam(0.0001)

model.compile(optimizer)

model.fit_generator(generator, epochs=10)

