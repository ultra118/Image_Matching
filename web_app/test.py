from keras.models import load_model
import cv2
import numpy as np
import os


model = load_model('C:/TakeMeHome/IM/dl_model/siamese_net_200_test.h5')

# image를 resize해주고 reshape.. 인자로 size를 줘야함
def resize_reshape_img(img_path, img_size):
    img = cv2.resize(cv2.imread(img_path),(img_size,img_size))
    return img.reshape([-1, img_size, img_size, 3])

img_path = 'C:/TakeMeHome/IM/dl_model/small_traiging_images2/'
original_img_path = 'C:/TakeMeHome/IM/dl_model/small_traiging_images2/emma_10_extend/hermi_1_box.jpg'
# original img <- 들어오는 이미지 shape과 size를 바꿔줌
original_img = resize_reshape_img(original_img_path, 200)

acc_list = []
file_list = []
# 전체경로에서 각 이미지 폴더에 접근
for folder in os.listdir(img_path):
    folder_path = os.path.join(img_path, folder)
    # 각 이미지 폴더에서 폴더내 이미지 접근
    for file in os.listdir(folder_path):
        print("{} predict...".format(file))
        file_path = os.path.join(folder_path, file)
        # db폴더내의 각 이미지에 대해 resize , reshape
        compare_file = resize_reshape_img(file_path, 200)
        file_list.append(str.split(file, '.')[0])
        acc_list.append(model.predict([original_img, compare_file])[0])
acc_np = np.array(acc_list)        
max_idx = np.argmax(acc_np)
print('file name : {}, acc : {}'.format(file_list[max_idx], acc_np[max_idx]))