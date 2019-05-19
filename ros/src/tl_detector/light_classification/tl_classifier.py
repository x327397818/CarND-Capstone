from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np

__FROZEN_SIMULATION_GRAPH_PATH__ = r'light_classification/sim_frozen_inference_graph.pb'
__FROZEN_REAL_WORLD_GRAPH_PATH__ = r'light_classification/site_frozen_inference_graph.pb'
__SCORE_THRESH__ = 0.5


class TLClassifier(object):
    """Classifier for traffic light.
    This module is learned from the following repository.
    https://github.com/alex-lechner/Traffic-Light-Classification.git

    It is using tensorflow provided SSR Inception V2 module, to train the labelled
    dataset on simulation dataset and real world dataset individually.
    """
    def __init__(self, is_site):
        # If it is site(non simulation)
        self.is_site = is_site

        # Frozen graph
        self.frozen_graph_path = None
        self.frozen_graph = tf.Graph()

        # Frozen graph saved information
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.num_detections = None
        self._get_frozen_graph_path()
        self._read_frozen_graph()

        self.session = tf.Session(graph=self.frozen_graph)

    def _get_frozen_graph_path(self):
        """
        Based on simulation/non-simulation, select corresponding graph to use.
        Then get get information from the graph
        """
        if self.is_site:
            self.frozen_graph_path = __FROZEN_REAL_WORLD_GRAPH_PATH__
        else:
            self.frozen_graph_path = __FROZEN_SIMULATION_GRAPH_PATH__

    def _read_frozen_graph(self):
        """
        Parse the tensor information from the frozen graph given
        """
        with self.frozen_graph.as_default():
            graph_definition = tf.GraphDef()
            with tf.gfile.GFile(self.frozen_graph_path, 'rb') as fid:
                graph_definition.ParseFromString(fid.read())
                tf.import_graph_def(graph_definition, name='')
            self.image_tensor = self.frozen_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.frozen_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.frozen_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.frozen_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.frozen_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        """
        with self.frozen_graph.as_default():
            (boxes, scores, classes, num_detections) = self.session.run(
                [self.detection_boxes,
                 self.detection_scores,
                 self.detection_classes,
                 self.num_detections],
                feed_dict={self.image_tensor: np.expand_dims(image, axis=0)})
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        if scores[0] > __SCORE_THRESH__:
            if classes[0] == 1:
                return TrafficLight.GREEN
            elif classes[0] == 2:
                return TrafficLight.RED
            elif classes[0] == 3:
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
