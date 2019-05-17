import face_recognition as fr
import cv2
from PIL import Image,ImageDraw

features = ['chin','left_eyebrow','right_eyebrow','nose_bridge','nose_tip','left_eye','right_eye','top_lip','bottom_lip']

def draw_face(im,loc):
    for face in loc:
        t, r, b, l = face
        cv2.rectangle(im, (l, t), (r, b), (0, 255, 0), 2)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Output", im_rgb)
    # cv2.waitKey(0)

def draw_landmark(im ,landmarks):
    pim = Image.fromarray(im)
    d = ImageDraw.Draw(pim)
    for lm in landmarks:
        for ft in features:
            d.line(lm[ft],width=3)
    pim.show()


im=fr.load_image_file('ceshi.bmp')
loc=fr.face_locations(im)
landmarks=fr.face_landmarks(im,loc)
print(loc)
print(landmarks)
draw_landmark(im,landmarks)

