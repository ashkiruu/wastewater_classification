"""
Export the trained Keras model to ONNX for the web app (Vercel deployment).

Run from the project root after (re)training:
    python scripts/export_onnx.py

Writes waste_classifier.onnx and checks it against the Keras model on the
sample images in static/manual_dataset.
"""
import glob
import os

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from PIL import Image

KERAS_PATH = "waste_classifier_mobilenetv2.keras"
ONNX_PATH = "waste_classifier.onnx"


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)


def main():
    model = tf.keras.models.load_model(KERAS_PATH, compile=False)
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=ONNX_PATH)
    print(f"Wrote {ONNX_PATH} ({os.path.getsize(ONNX_PATH) / 1e6:.1f} MB)")

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    files = sorted(glob.glob("static/manual_dataset/*"))
    agree, worst = 0, 0.0
    for f in files:
        x = preprocess(f)
        a = model.predict(x, verbose=0)[0]
        b = sess.run(None, {name: x})[0][0]
        agree += int(a.argmax() == b.argmax())
        worst = max(worst, float(np.abs(a - b).max()))
    print(f"Parity check: {agree}/{len(files)} same top-1, max prob diff {worst:.2e}")


if __name__ == "__main__":
    main()
