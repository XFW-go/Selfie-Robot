import face_recognition as fr
import os
import cv2
import numpy as np
from PIL import Image

path = './imgs/'
dest = './dest_imgs/'
files = os.listdir(path)

ratio_face_field = 0.4
ratio_face_whole = 0.2
alpha = 1
beita = 0.5
w_img = 800
h_img = 600
S_img = 800*600
theta = 0.005

def evaluate(areas, ct, bound, mode=0, bias=0, size=1):
    # 超参1 人脸均值偏向性，左中右
    # 超参2 是否要对脸的大小进行限制
    score = 0
    S_faces = sum(areas)
    ct_x,ct_y=0.,0.
    num = len(areas)
    compact_score,size_score,center_score=0,0,0
    if mode == 0: # More than 4 people in the image
        # faces should be compact
        S_face_field = (bound[2]-bound[0])*(bound[3]-bound[1])
        rt = S_faces/S_face_field
        if rt > ratio_face_field:
            compact_score = alpha * rt
        else:
            compact_score = beita * rt
        # face size should not be too large
        if size == 1:
            rt_w = S_faces/S_img
            if  rt_w > ratio_face_whole:
                size_score = 0.5 - 0.8*rt_w
            else:
                size_score = 0.5 - 0.5*rt_w
        else:
            size_score = 0.4
        # faces should be round the center of the pic
        ct_x += sum(ct[:,0])
        ct_y += sum(ct[:,1])
        ct_x /= num; ct_y /= num

        t = pow(ct_x-w_img/2,2)+0.1*pow(ct_y-h_img/2,2)
        center_score = 1-theta*np.sqrt(t)

    else: # At most 4 people
        # faces should not be too far from each other
        if num>1:
            dis=0
            for i in range(1,num):
                dis+=(ct[i][0]-ct[i-1][0])+0.3*(ct[i][1]-ct[i-1][1])
            dis=dis/(num-1)
            compact_score=dis
            if compact_score>350:
                print('too far from each other')
        # face size should not be too large or too small
        if size == 1:
            rt_w = S_faces/S_img
            #print(rt_w)
            if  rt_w > ratio_face_whole:
                size_score = 0.8 - 1*rt_w
            else:
                size_score = 0.8 - 0.7*rt_w
            if rt_w<0.1:
                print('heads are too small')
            elif rt_w>0.25:
                print('heads are too big')
        else:
            size_score= 0.6

        # faces should be round the center of the pic
        ct_x += sum(ct[:,0])
        ct_y += sum(ct[:,1])
        ct_x /= num; ct_y /= num
        #print('x:{:.2f},y:{:.2f}'.format(ct_x, ct_y))
        t = pow(ct_x-w_img/2,2)+0.1*pow(ct_y-h_img/2,2)
        center_score = 1-theta*np.sqrt(t)
        if center_score<0.6:
            if ct_x<350:
                print('too left')
            elif ct_x>450:
                print('too right')
            if ct_y<200:
                print('too high')
            elif ct_y>400:
                print('too low')
    return compact_score,size_score,center_score


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
    ct=np.array(ct)
    ct=ct[ct[:,0].argsort()]
    return im_rgb, np.array(area), ct, np.array(bound)


for file in files:
    img = fr.load_image_file(path+file)
    img1 = cv2.imread(path+file)
    loc = fr.face_locations(img)
    res, areas, centers, bound = parse_face(img, loc)
    #for i in range(len(areas)):
    #    print('areas:{} center:{} bound:{}'.format(areas[i],centers[i],bound[i]))
    #cv2.imwrite(dest+file, res)
    coms,sizes,cens=evaluate(areas, centers, bound, mode=1)
    print('file:{}, compact:{:.3f},size:{:.3f},center:{:.3f}'.format(file,coms,sizes,cens))



