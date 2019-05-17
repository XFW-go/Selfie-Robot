import face_recognition as fr
import os
import cv2
import numpy as np
from PIL import Image

path = './select_imgs/'
dest = './dest_imgs/'
files = os.listdir(path)

ratio_face_field = 0.4
ratio_face_whole = 0.2
alpha = 1
beita = 0.5
w_img = 800
h_img = 600
S_img = 800*600
theta = 0.001

def evaluate(areas, ct, bound, mode=0, bias=0, size=1):
	# 超参1 人脸均值偏向性，左中右
	# 超参2 是否要对脸的大小进行限制
	score = 0.
	S_face_field = 0.
	S_faces = 0.
	ct_x=0.; ct_y=0.
	num = len(areas)
	if mode == 0: # More than 4 people in the image
		# faces should be compact
		S_faces += sum(areas)
		S_face_field = (bound[2]-bound[0])*(bound[3]-bound[1])
		rt = S_faces/S_face_field
		if rt > ratio_face_field:
			score += alpha * rt
		else:
			score += belta * rt
		
		# face size should not be too large
		if size == 1:
			rt_w = S_faces/S_img
			if  rt_w > ratio_face_whole:
				score += 0.5 - 0.8*rt_w
			else:
				score += 0.5 - 0.5*rt_w
		else:
			score += 0.4
		
		# faces should be round the center of the pic
		ct_x += sum(ct[:][0])
		ct_y += sum(ct[:][1])
		ct_x /= num; ct_y /= num
		t = pow(ct_x-w_img/2,2)+0.1*pow(ct_y-h_img/2,2)
		score += 1-theta*np.sqrt(t)
		
	else: # At most 4 people
		# face size should not be too large
		if size == 1:
			rt_w = S_faces/S_img
			if  rt_w > ratio_face_whole:
				score += 0.7 - 1*rt_w
			else:
				score += 0.7 - 0.7*rt_w
		else:
			score += 0.6
		
		# faces should be round the center of the pic
		ct_x += sum(ct[:][0])
		ct_y += sum(ct[:][1])
		ct_x /= num; ct_y /= num
		t = pow(ct_x-w_img/2,2)+0.1*pow(ct_y-h_img/2,2)
		score += 1-theta*np.sqrt(t)			
	
	return score

def parse_face(im,loc):
	area = []
	ct = []
	bound = []
	for face in loc:
		y0, x1, y1, x0 = face
		ml=10000; mt=10000; mr=0; md=0; # Bound of the face area
		coord = []; bd = []
		cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 2)
		# compute area
		w = x1-x0+0.0
		h = y1-y0+0.0
		s = w * h
		# set bound
		if x0<ml: ml=x0;
		if y0<mt: mt=y0;
		if x1>mr: mr=x1;
		if y1>md: md=y1;
		area.append(s)
		coord = [(x0+x1)/2.0, (y0+y1)/2.0]
		bd = [ml,mt,mr,md]
		ct.append(coord)
		bound.append(bd)
	im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	return im_rgb, area, ct, bound


for file in files:
	img = fr.load_image_file(path+file)
	img1 = cv2.imread(path+file)
	loc = fr.face_locations(img)
	res, areas, centers, bound = parse_face(img, loc)
	#cv2.imwrite(dest+file, res)
	print(evaluate(areas, centers, bound, 1))
	


