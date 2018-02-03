import keras_rcnn
from keras_rcnn import backend, datasets, layers, models, preprocessing
import numpy
import cv2
import sys
import matplotlib
import keras
import json
import rcnn_utils
import skimage

train_json_file = '/home/paperspace/bowl/input/DSB208_train.json'
# test_json_file = '/home/paperspace/bowl/DSB208_test.json'
train_path = '/home/paperspace/bowl/input/stage1_train/'

img_size=256
batch_size=1

# training, test = datasets.load_data('DSB2018')

# training, validation = sklearn.model_selection.train_test_split(training)

with open(train_json_file, 'r') as file:
	training = json.loads(file.read())

# with open(test_json_file, 'r') as file:
# 	test = json.loads(file.read())

classes = {
	"nucleus": 1
}

# print(type(training))
# print(training[0])
# print(training[0].keys())

# for item in test:
# 	item['shape'] = (item['image']['shape']['r'], item['image']['shape']['c'], item['image']['shape']['channels'])
# 	item['filename'] = item['image']['pathname']
	#del item['image']['shape']
	#del item['image']['pathname']

for item in training:
	#item['shape'] = (item['image']['shape']['r'], item['image']['shape']['c'], item['image']['shape']['channels'])
	#item['filename'] = item['image']['pathname']
	#item['boxes'] = []
	#for x in item['objects']:
		#item['boxes'].append({})
		#item['boxes'][-1]['class'] = x['class']
		#item['boxes'].append([x['bounding_box']['minimum']['c'], x['bounding_box']['minimum']['r'], 
		#			 x['bounding_box']['maximum']['c'], x['bounding_box']['maximum']['r']])
		#item['boxes'][-1]['x1'] = x['bounding_box']['minimum']['c']
		#item['boxes'][-1]['x2'] = x['bounding_box']['maximum']['c']
		#item['boxes'][-1]['y1'] = x['bounding_box']['minimum']['r']
		#item['boxes'][-1]['y2'] = x['bounding_box']['maximum']['r']
	# item['boxes'] = []
	# for x in item['objects']:

		# item['boxes'].append([x['bounding_box']['minimum']['c'], x['bounding_box']['minimum']['r'], 
		# 			 x['bounding_box']['maximum']['c'], x['bounding_box']['maximum']['r']])
		
	item['boxes'] = numpy.array(item['boxes'])
	item['class'] = numpy.array([[0,1] for x in range(len(item['boxes']))])

print('loading data...')
# training = rcnn_utils.make_json(train_path, img_size)

# with open(train_json_file, 'w') as file:
# 	json.dump(training, file)
# sys.exit()
# with open(test_json_file, 'w') as file:
# 	json.dump(test, file)

# generator = preprocessing.ObjectDetectionGenerator()
# generator = preprocessing.ImageSegmentationGenerator()
# generator = generator.flow(training, classes, target_shape=(img_size, img_size), scale=1.0, batch_size=batch_size, ox=0, oy=0)

# val_generator = preprocessing.ObjectDetectionGenerator()

# val_generator = generator.flow(validation, classes, target_shape=(img_size, img_size), scale=1.0, batch_size=batch_size)

class train_gen:
	def __init__(self, training):
		self.training = training
	def __iter__(self):
		while True:
			for item in self.training:
				target_image = numpy.expand_dims(skimage.io.imread(item['filename']), 0).astype(keras.backend.floatx())
				target_bounding_boxes = numpy.expand_dims(item['boxes'], 0).astype(numpy.float64)
				for i in range(0, target_bounding_boxes.shape[1]):
					if target_bounding_boxes[:,i,2] == target_image.shape[2]:
						target_bounding_boxes[:,i,2] -= 1
					if target_bounding_boxes[:,i,3] == target_image.shape[1]:
						target_bounding_boxes[:,i,3] -= 1
					if target_bounding_boxes[:,i,2] > target_image.shape[2]:
						print('uh oh')
				#target_bounding_boxes = numpy.reshape(target_bounding_boxes, (-1, 0, 4))
				target_scores = numpy.expand_dims(item['class'], 0).astype(numpy.int64)
				#target_scores = numpy.reshape(target_scores, (-1, 0, 2))
				# print(target_bounding_boxes.shape)
				metadata = numpy.array([[target_image.shape[2], target_image.shape[1], 1.0]])
				#print(metadata.shape)
				result = [target_bounding_boxes, target_image, target_scores, metadata], None
				# print(result)
				yield result
# 	def next(self):
# 		return next(self.generator)
generator = iter(train_gen(training))

# generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

# generator = generator.flow(training, classes)

# for _ in range(0,1):
# 	(target_bounding_boxes, target_image, target_scores, meta), _ = next(generator)

# 	print(type(target_bounding_boxes))
# 	print(target_bounding_boxes.dtype)
# 	print(target_bounding_boxes)
# 	print(type(target_image))
# 	print(target_image.dtype)
# 	print(target_image.shape)
# 	print(type(target_scores))
# 	print(target_scores.dtype)
# 	print(target_scores.shape)
# 	print(type(meta))
# 	print(meta.dtype)
# 	print(meta.shape)
# 	print(meta)

	
# 	target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

# 	target_image = numpy.squeeze(target_image)

# 	target_scores = numpy.argmax(target_scores, -1)

# 	target_scores = numpy.squeeze(target_scores)
	
# 	#print(target_bounding_boxes.shape)
# 	#print(target_image.shape)
# 	#print(target_scores.shape)
	
# 	_, axis = matplotlib.pyplot.subplots(1, figsize=(12, 8))

# 	axis.imshow(target_image)

# 	for target_index, target_score in enumerate(target_scores):
# 		if target_score > 0:
# 			xy = [
# 				target_bounding_boxes[target_index][0],
# 				target_bounding_boxes[target_index][1]
# 			]
# 			#print(xy)
# 			w = target_bounding_boxes[target_index][2] - target_bounding_boxes[target_index][0]
# 			h = target_bounding_boxes[target_index][3] - target_bounding_boxes[target_index][1]
# 			rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")
# 			#print(xy, w, h)
# 			axis.add_patch(rectangle)

# 	matplotlib.pyplot.show()

# sys.exit()




# for i in range(0, 5):
# 	target_image = skimage.io.imread(training[i]['filename'])[:,:,:3]
# 	target_bounding_boxes = training[i]['boxes']
# 	target_scores = training[i]['class']
# 	#print('loading one image')
# 	#(target_bounding_boxes, target_image, target_scores, _), _ = generator.next()
# 	#print('loaded one image')
# 	#target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

# 	#target_image = numpy.squeeze(target_image)

# 	# target_scores = numpy.argmax(target_scores, -1)

# 	# target_scores = numpy.squeeze(target_scores)

# 	_, axis = matplotlib.pyplot.subplots(1, figsize=(12, 8))

# 	axis.imshow(target_image)

# 	#for target_index, target_score in enumerate(target_scores):
# 	for i, box in enumerate(target_bounding_boxes):
# 		#if target_score > 0:
# 		target_score = target_scores[i]
# 		xy = [
# 			box['x1'],
# 			box['y1']
# 		]

# 		w = box['x2'] - box['x1']
# 		h = box['y2'] - box['y1']

# 		rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")

# 		axis.add_patch(rectangle)

# 	matplotlib.pyplot.show()

print('building model...')
model_input = keras.layers.Input((None, None, 3))

model = keras_rcnn.models.RCNN(model_input, classes=len(classes) + 1)

optimizer = keras.optimizers.Adam(0.0001)

model.compile(optimizer)
model.summary(line_length=150)
model.fit_generator(generator, epochs=1, steps_per_epoch=len(training)/10)

# visualize prediction
example, _ = next(generator)
target_bounding_boxes, target_image, target_labels, _ = example
target_bounding_boxes = numpy.squeeze(target_bounding_boxes)
target_image = numpy.squeeze(target_image)
target_labels = numpy.argmax(target_labels, -1)
target_labels = numpy.squeeze(target_labels)
# output_anchors, output_proposals, output_deltas, output_scores = model.predict(example)
output_anchors, output_scores = model.predict(example)
output_anchors = numpy.squeeze(output_anchors)
# output_proposals = numpy.squeeze(output_proposals)
# output_deltas = numpy.squeeze(output_deltas)
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

