import keras_rcnn
from keras_rcnn import backend, datasets, layers, models, preprocessing
import numpy
import matplotlib
import keras
import json

train_json_file = '/home/paperspace/bowl/input/DSB208_train.json'
test_json_file = '/home/paperspace/bowl/input/DSB208_test.json'
img_size=256
batch_size=1

training, test = datasets.load_data('DSB2018')

# training, validation = sklearn.model_selection.train_test_split(training)

classes = {
    "nucleus": 1
}

# print(type(training))
# print(training[0])

for item in test:
	item['shape'] = (item['image']['shape']['r'], item['image']['shape']['c'], item['image']['shape']['channels'])
	item['filename'] = item['image']['pathname']

for item in training:
	item['shape'] = (item['image']['shape']['r'], item['image']['shape']['c'], item['image']['shape']['channels'])
	item['filename'] = item['image']['pathname']
	item['boxes'] = item['objects']
	for x in item['boxes']:
		x['x1'] = x['bounding_box']['minimum']['c']
		x['x2'] = x['bounding_box']['maximum']['c']
		x['y1'] = x['bounding_box']['minimum']['r']
		x['y2'] = x['bounding_box']['maximum']['r']

with open(train_json_file, 'w') as file:
	json.dump(training, file)

with open(test_json_file, 'w') as file:
	json.dump(test, file)
	
sys.exit()

with open(train_json_file, 'r') as file:
	training = json.loads(file.read())

with open(test_json_file, 'r') as file:
	test = json.loads(file.read())

generator = preprocessing.ObjectDetectionGenerator()

generator = generator.flow(training, classes, target_shape=(img_size, img_size), scale=1.0, batch_size=batch_size)

# val_generator = preprocessing.ObjectDetectionGenerator()

# val_generator = generator.flow(validation, classes, target_shape=(img_size, img_size), scale=1.0, batch_size=batch_size)


# for x in range(0, 5):
# 	(target_bounding_boxes, target_image, target_scores, _), _ = generator.next()

# 	target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

# 	target_image = numpy.squeeze(target_image)

# 	target_scores = numpy.argmax(target_scores, -1)

# 	target_scores = numpy.squeeze(target_scores)

# 	_, axis = matplotlib.pyplot.subplots(1, figsize=(12, 8))

# 	axis.imshow(target_image)

# 	for target_index, target_score in enumerate(target_scores):
# 	    if target_score > 0:
# 	        xy = [
# 	            target_bounding_boxes[target_index][0],
# 	            target_bounding_boxes[target_index][1]
# 	        ]

# 	        w = target_bounding_boxes[target_index][2] - target_bounding_boxes[target_index][0]
# 	        h = target_bounding_boxes[target_index][3] - target_bounding_boxes[target_index][1]

# 	        rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")

# 	        axis.add_patch(rectangle)

# 	matplotlib.pyplot.show()

image = keras.layers.Input((img_size, img_size, 3))

model = keras_rcnn.models.RCNN(image, classes=len(classes) + 1)

optimizer = keras.optimizers.Adam(0.0001)

model.compile(optimizer)

model.fit_generator(generator, epochs=1)

# visualize prediction
example, _ = generator.next()
target_bounding_boxes, target_image, target_labels, _ = example
target_bounding_boxes = numpy.squeeze(target_bounding_boxes)
target_image = numpy.squeeze(target_image)
target_labels = numpy.argmax(target_labels, -1)
target_labels = numpy.squeeze(target_labels)
output_anchors, output_proposals, output_deltas, output_scores = model.predict(example)
output_anchors = numpy.squeeze(output_anchors)
output_proposals = numpy.squeeze(output_proposals)
output_deltas = numpy.squeeze(output_deltas)
output_scores = numpy.squeeze(output_scores)
_, axis = matplotlib.pyplot.subplots(1)
axis.imshow(target_image)
for index, label in enumerate(target_labels):
	if label == 1:
	    xy = [
		target_bounding_boxes[index][0],
		target_bounding_boxes[index][1]
	    ]
	    w = target_bounding_boxes[index][2] - target_bounding_boxes[index][0]
	    h = target_bounding_boxes[index][3] - target_bounding_boxes[index][1]
	    rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="g", facecolor="none")
	    axis.add_patch(rectangle)
for index, score in enumerate(output_scores):
	if score > 0.95:
	    xy = [
		output_anchors[index][0],
		output_anchors[index][1]
	    ]
	    w = output_anchors[index][2] - output_anchors[index][0]
	    h = output_anchors[index][3] - output_anchors[index][1]
	    rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")
	    axis.add_patch(rectangle)
matplotlib.pyplot.show()

