import onnx
import onnxruntime
import cv2
import numpy as np

# load onnx model
onnx_model = onnx.load("DAVE_NVIDIA_1_PGD_v2.onnx")

# chose one img to compare the predict result:
full_image = cv2.imread(
    "driving_dataset/" + str(0) + ".jpg")
image = cv2.resize(full_image[-150:], (200, 66)) / 255.0


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from onnx2keras import onnx_to_keras

# # reconstruct keras model
keras_model = onnx_to_keras(onnx_model, ['input'])
keras_model.summary()

# import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# execute by keras
x = image[None, :, :, :]
print("origin shape")
print(x.shape)
# if tf.test.is_built_with_cuda():
tf.keras.backend.set_image_data_format('channels_last')
# temp_x_train = []
# for i in range(len(x_train)):
    # new_x_train_row=np.moveaxis(x_train[i],0,2)
    # temp_x_train.append(new_x_train_row)
# x_train = np.array(temp_x_train)
x = tf.transpose(x, [0, 3, 1, 2])
# x = tf.transpose(x, [0, 3, 1, 2])
print(x.shape)
degrees1 = float(keras_model.predict(x, batch_size=1, steps=1))
print(degrees1)
# degrees1 = 0.1

# execute by onnx
x = image[None, :, :, :]
content = onnx_model.SerializeToString()
sess = onnxruntime.InferenceSession(content)
x = x if isinstance(x, list) else [x]
float32_x = np.array(x).astype(np.float32)  # onnx model accepts float32 input
print(float32_x.shape)
float32_x = float32_x.transpose([0, 1, 4, 2, 3])
feed = dict([(input.name, float32_x[n])
             for n, input in enumerate(sess.get_inputs())])
pred_onnx = sess.run(None, feed)
degrees2 = pred_onnx[0]

# compare execution results
print("predictions by onnx/keras: %.3f / %.3f " %
      (degrees2, degrees1))
