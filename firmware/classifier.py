import numpy as np

from rules import LABELS

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter


class Classifier:
    def __init__(self, path):
        self.interpreter = Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.size = int(self.input["shape"][1])

    def predict(self, image):
        frame = np.asarray(image, dtype=np.uint8)
        if frame.shape[0] != self.size or frame.shape[1] != self.size:
            raise ValueError("expected " + str(self.size) + " px square, got " + str(frame.shape))

        self.interpreter.set_tensor(self.input["index"], np.expand_dims(frame, axis=0))
        self.interpreter.invoke()

        scores = self.interpreter.get_tensor(self.output["index"])[0].astype(np.float32)
        scale, zero = self.output["quantization"]
        if scale:
            scores = (scores - zero) * scale

        exp = np.exp(scores - scores.max())
        probs = exp / exp.sum()
        index = int(probs.argmax())
        return LABELS[index], float(probs[index])
