import tensorflow_hub as hub
# Apply image detector on a single image.
detector = hub.load("https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/TensorFlow2/fpnlite-320x320/1")
detector_output = detector(image_tensor)
class_ids = detector_output["detection_classes"]