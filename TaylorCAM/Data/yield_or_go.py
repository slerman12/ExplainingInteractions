import argparse
import cv2
import os
import numpy as np
import random
import pickle

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train-size', type=int, default=9800)
parser.add_argument('--test-size', type=int, default=200)
parser.add_argument('--data-dir', type=str, default="./")
args = parser.parse_args()

train_size = args.train_size
test_size = args.test_size
img_size = 300
size = 20
question_size = 11 ##6 for one-hot vector of color, 2 for question type, 3 for question subtype
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""

nb_questions = 10
dirs = args.data_dir

colors = [
    (167, 65, 74),  # re
    (40, 39, 38),  # blac
    (106, 138, 130),  # gr
    (163, 124, 39),  # ye
    (86, 56, 56),  # br
    (105, 117, 166)   # pu
]


try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))

def center_generate(objects):
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)        
        if len(objects) > 0:
            for name,c,shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center



def build_dataset():
    objects = []
    img = np.ones((img_size,img_size,3)) * 255
    yield_sign = False
    passing_car = False
    for color_id,color in enumerate(colors):  
        center = center_generate(objects)
        ran = 0.32
        if color_id == 0:
            ran = .32
        if random.random()<ran:
            start = (center[0]-size+size//4+size//8, center[1]-size+size//8)
            end = (center[0]+size-size//4-size//8, center[1]+size-size//8)
            cv2.rectangle(img, start, end, color, -1)
            objects.append((color_id,center,'c'))
            passing_car = True
        else:
            p1 = [center[0]-size + size//6, center[1]-size + size//6]
            p2 = [center[0]-size + size//6, center[1]+size - size//6]
            p3 = [center[0] + size - size//6, center[1]]
            # draw a triangle
            vertices = np.array([p1, p2, p3], np.int32)
            pts = vertices.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(234, 231, 175), thickness=size//6)


            # fill it
            cv2.fillPoly(img, [pts], color=color)
            objects.append((color_id,center,'y'))
            if color_id == 0:
                yield_sign = True

    rel_answers = 0
    if passing_car and yield_sign:
        rel_answers = 1

    img = img/255.
    dataset = (img, rel_answers)
    return dataset


print('building test datasets...')
test_datasets = [build_dataset() for _ in range(test_size)]
print('building train datasets...')
train_datasets = [build_dataset() for _ in range(train_size)]


#img_count = 0
#cv2.imwrite(os.path.join(dirs,'{}.png'.format(img_count)), cv2.resize(train_datasets[0][0]*255, (512,512)))


print('saving datasets...')
filename = os.path.join(dirs, 'yield-or-go.pickle')
with open(filename, 'wb') as f:
    pickle.dump((train_datasets, test_datasets), f)
print('datasets saved at {}'.format(filename))
