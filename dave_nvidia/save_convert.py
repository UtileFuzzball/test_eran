# import onnx
# import onnxruntime
import cv2
import numpy as np
import os

# load onnx model
# onnx_model = onnx.load("DAVE_NVIDIA_1_PGD_v2.onnx")

def dir_path(dirname):
	for maindir,subdir,file_name_list in os.walk(dirname):
		print(maindir)
		print(subdir)
		print(file_name_list)

		for filename in file_name_list:
			print(filename)
	return maindir,file_name_list
# dir_path("/home/wangsiqi/w77/sundries/dx/ERAN/dave_nvidia/driving_dataset")

def convert(maindir,file_name_list):
	for filename in file_name_list:
		print(maindir+filename)
		full_image = cv2.imread(maindir+"/"+filename)
		image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
		x = image[None, :, :, :]
		x = x if isinstance(x, list) else [x]
		float32_x = np.array(x).astype(np.float32)  # onnx model accepts float32 input
		print(float32_x.shape)
		float32_x = float32_x.transpose([0, 1, 4, 2, 3])
		print(float32_x.shape)
maindir,file_name_list=dir_path("/home/wangsiqi/w77/sundries/dx/ERAN/dave_nvidia/driving_dataset")
convert(maindir,file_name_list)