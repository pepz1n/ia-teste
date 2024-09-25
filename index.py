import tensorflow as tf
import tensorflow_hub as hub
import cv2

model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
detector = hub.load(model_url)

def load_labels(label_map_path):
    with open(label_map_path, 'r') as file:
        label_map = {}
        for line in file.readlines():
            if 'name:' in line:
                name = line.split(': ')[1].strip().replace("'", "")
                label_map[int(line.split(':')[0].split(' ')[-1])] = name
    return label_map

# label_map = load_labels('label_map.pbtxt')

def detect_objects(frame):
    img_tensor = tf.convert_to_tensor(frame)
    img_tensor = img_tensor[tf.newaxis, ...]
    detections = detector(img_tensor)
    return detections

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_image = cv2.resize(frame, (300, 300))

    detections = detect_objects(input_image)

    num_detections = int(detections["detection_scores"].shape[1])
    for i in range(num_detections):
        score = int(detections["detection_scores"][0, i].numpy() * 100)
        name = detections["detection_classes"][0, i].numpy()
        class_id = detections["detection_classes"]
        ind  = detections["detection_anchor_indices"]
        bbox = detections["detection_boxes"][0, i].numpy()
        print(ind)
        if score > 50 and name == 1:
            h, w, _ = frame.shape
            y_min, x_min, y_max, x_max = bbox
            x_min, x_max, y_min, y_max = int(x_min * w), int(x_max * w), int(y_min * h), int(y_max * h)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            # label = label_map.get(name, "Unknown")
            cv2.putText(frame, f'Score: {score}% Person ', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)


    cv2.imshow('Detecção de Pessoas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
