# import onnx
import csv
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
		# x=np.reshape(full_image)
		print(full_image.shape)
		full_image=cv2.resize(full_image,(128,32))
		print(full_image.shape)
		# print(full_image)
		filename="new_test_full.csv"
		writecsv(filename, full_image)


def writecsv(filename,image):
	with open(filename,"a+",newline="")as file:
		# csvwriter=csv.writer(datacsv)
		file.write("0,")
		for i in range(len(image)):
			# print(image[i])
			for j in range(len(image[i])):
				for k in range(len(image[i][j])):
					file.write(str(image[i][j][k]))
					if i==len(image)-1 and j==len(image[i])-1 and k==len(image[i][j])-1:
						file.write("\n")
					else:
						file.write(",")

	pass
maindir,file_name_list=dir_path("/home/wangsiqi/w77/sundries/dx/ERAN/data/examples_to_run/IMG/")
convert(maindir,file_name_list)