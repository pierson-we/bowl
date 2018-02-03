import keras_rcnn
from keras_rcnn import backend, datasets, layers, models, preprocessing
import numpy
import matplotlib
import keras
import json
import rcnn_utils
import skimage

train_json_file = '/home/paperspace/bowl/DSB208_train.json'
# test_json_file = '/home/paperspace/bowl/DSB208_test.json'
train_path = '/home/paperspace/bowl/input/stage1_train/'

img_size=256
batch_size=1

# training, test = datasets.load_data('DSB2018')

# training, validation = sklearn.model_selection.train_test_split(training)

classes = {
    "nucleus": 1
}

with open(train_json_file, 'r') as file:
	training = json.load(file.read())

# with open(test_json_file, 'r') as file:
# 	test = json.loads(file.read())

# print(type(training))
# print(training[0])
# print(training[0].keys())

# for item in test:
# 	item['shape'] = (item['image']['shape']['r'], item['image']['shape']['c'], item['image']['shape']['channels'])
# 	item['filename'] = item['image']['pathname']
# 	del item['image']['shape']
# 	del item['image']['pathname']

# for item in training:
# 	del item['image']
# 	item['shape'] = (item['image']['shape']['r'], item['image']['shape']['c'], item['image']['shape']['channels'])
# 	item['filename'] = item['image']['pathname']
# 	item['boxes'] = []
# 	for x in item['objects']:
# 		item['boxes'].append({})
# 		item['boxes'][-1]['class'] = x['class']
# 		item['boxes'][-1]['x1'] = x['bounding_box']['minimum']['c']
# 		item['boxes'][-1]['x2'] = x['bounding_box']['maximum']['c']
# 		item['boxes'][-1]['y1'] = x['bounding_box']['minimum']['r']
# 		item['boxes'][-1]['y2'] = x['bounding_box']['maximum']['r']
# 	del item['image']['shape']
# 	del item['image']['pathname']
# 	del item['objects']

print('loading data...')
# training = rcnn_utils.make_json(train_path, img_size)

# with open(train_json_file, 'w') as file:
# 	json.dump(training, file)

# with open(test_json_file, 'w') as file:
# 	json.dump(test, file)

generator = preprocessing.ObjectDetectionGenerator()
# generator = preprocessing.ImageSegmentationGenerator()
generator = generator.flow(training, classes, target_shape=(img_size, img_size), scale=1.0, batch_size=batch_size, ox=0, oy=0)

# val_generator = preprocessing.ObjectDetectionGenerator()

# val_generator = generator.flow(validation, classes, target_shape=(img_size, img_size), scale=1.0, batch_size=batch_size)


for i in range(0, 5):
	target_image = skimage.io.imread(training[i]['filename'])[:,:,:3]
	target_bounding_boxes = training[i]['boxes']
	
	#print('loading one image')
	#(target_bounding_boxes, target_image, target_scores, _), _ = generator.next()
	#print('loaded one image')
	#target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

	#target_image = numpy.squeeze(target_image)

	# target_scores = numpy.argmax(target_scores, -1)

	# target_scores = numpy.squeeze(target_scores)

	_, axis = matplotlib.pyplot.subplots(1, figsize=(12, 8))

	axis.imshow(target_image)

	#for target_index, target_score in enumerate(target_scores):
	for box in target_bounding_boxes:
		#if target_score > 0:
		target_score = box['class']
		xy = [
			box['x1'],
			box['y1']
		]

		w = box['x2'] - box['x1']
		h = box['y2'] - box['y1']

		rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")

		axis.add_patch(rectangle)

	matplotlib.pyplot.show()

print('building model...')
image = keras.layers.Input((img_size, img_size, 4))

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

