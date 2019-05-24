import face_recognition
import cv2
import os
import numpy as np

class Image_Preprocessing:
    def __init__(self):
        pass
    def face_preprocessing(self, image_path):
        return self.crop_rotate(image_path)

    # 얼굴 영역별로 각 사진을 회전시키고
    # 회전된 얼굴만 잘라냄(랜드마크와 얼굴검출영역 최대값으로 얼굴이 최대한 온전히 나오게)
    def crop_rotate(self, img_path):
        image = cv2.imread(img_path)
        face_landmarks = face_recognition.face_landmarks(image)

        image_list = []
        # 각 얼굴들 검출해서 회전시킬 중심점들을 찾아서 (몇번째 얼굴, 회전 중심)의 dict형태로 추가
        for landmarks in face_landmarks:
            left_eye_center = np.array(landmarks['left_eye']).mean(axis=0)
            right_eye_center = np.array(landmarks['right_eye']).mean(axis=0)
            lec_x = int(left_eye_center[0])
            lec_y = int(left_eye_center[1])
            rec_x = int(right_eye_center[0])
            rec_y = int(right_eye_center[1])
            # 눈의 중앙 좌표와 각을 구함
            angle = np.degrees(np.arctan2((lec_y - rec_y), (lec_x - rec_x))) - 180
            # print("left_eye_center : {} right_eye_center : {}".format((lec_x, lec_y), (rec_x, rec_y)))
            # print("angle : {}".format(angle))
            # 회전 중심 점
            center_rotate = [int((lec_x + rec_x) / 2), int((lec_y + rec_y) / 2)]
            M = cv2.getRotationMatrix2D((center_rotate[0], center_rotate[1]), angle, 1)
            image_list.append(self.crop_image(image, center_rotate, M))

        return image_list

    # 두 점 사이의 거리
    def distacne_two_point(self, list_point1, list_point2):
        dx = list_point1[0] - list_point2[0]
        dy = list_point1[1] - list_point2[1]
        # 굳이 소수점까지 볼 필요는 없음
        return int((dx ** 2 + dy ** 2) ** 0.5)

    # 각 이미지로 부터 회전 중심값과 회전 설정 값을 받아
    # 이미지를 회전시키고 그 중 보내진 이미지의 얼굴영역과 가장 가까운 얼굴을 도려냄
    def crop_image(self, image, center_rotate, M):
        rotate_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        face_landmarks = face_recognition.face_landmarks(rotate_image)
        distance_list = []

        for landmark in face_landmarks:
            left_eye_center = np.array(landmark['left_eye']).mean(axis=0)
            right_eye_center = np.array(landmark['right_eye']).mean(axis=0)
            eye_center = ((left_eye_center + right_eye_center) / 2).astype('uint16')
            # 회전된 이미지의 축은 이전 함수에서 해당하는 얼굴의 중심점이란걸 알 수 있다
            # 이 점과 그리고 현재 회전된 이미지에서의 landmark의 중심점 사이의 거리가 가장 짧은 얼굴이
            # 해당하는 얼굴이니 이 얼굴을 도려내야함
            # print("center_rotate : {} , eye_center : {}".format(center_rotate, eye_center))
            distance_list.append(self.distacne_two_point(eye_center, center_rotate))

        # distance가 가장 짧은 index를 가지고 옴
        for i, d in enumerate(distance_list):
            if d == min(distance_list):
                crop_image_index = i
        # print("distance_list : {}".format(distance_list))
        # print("crop_image_index : {}".format(crop_image_index))
        # print("len(face_landmarks) : {}".format(len(face_landmarks)))
        current_face_mark = face_landmarks[crop_image_index]

        landmarks_list = []
        for k in current_face_mark.keys():
            landmarks_list.extend(current_face_mark[k])
        landmarks_array = np.array(landmarks_list)
        rect_list = []
        # 얼굴 rect를 x,y좌표로 구분해놓고
        for array in np.split(np.array(face_recognition.face_locations(rotate_image)[crop_image_index]), 2):
            rect_list.append(array)
        # landmkar와 rect에 대한 x,y,좌표를 np.array로 만들어서 최소값 최대값을 쉽게 뽑도록 함
        for i in range(len(rect_list)):
            temp = rect_list[i][0]
            rect_list[i][0] = rect_list[i][1]
            rect_list[i][1] = temp
            landmarks_array = np.vstack((landmarks_array, rect_list[i].reshape((-1, 2))))
        min_xy = landmarks_array.min(axis=0)
        max_xy = landmarks_array.max(axis=0)

        return image[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]]


class Face_Embedding:
    # 생성자에서 db 만듬
    def __init__(self):
        self.create_db()
        self.img_preprocessing = Image_Preprocessing()

    def create_db(self):
        self.face_database = {}

    # image가 얼굴을 갖고 있는지 check
    def check_img(self, image_path):
        img_ = cv2.imread(image_path)
        if len(face_recognition.face_locations(img_)) == 0:
            return False
        return True

    # opencv의 경우는 rgb가아닌 bgr의 color이기 때문 plt로 출력할거 아니면 안바꿔도 됨
    def convert_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # iamge path로 image를 각 얼굴에대해 회전하고 랜드마크를 포함하게 잘라내서 embedding 값 받음
    def get_face_embedding(self, image_path):
        # 얼굴 없으면 0,0
        # print(image_path)
        if not self.check_img(image_path):
            print('얼굴을 찾을 수 없습니다.')
            return 0
        face_encoding_list = []
        images = self.img_preprocessing.face_preprocessing(image_path)
        for img in images:
            img_size = [(0, img.shape[1], img.shape[0], 0)]
            face_encoding = face_recognition.face_encodings(img, img_size)
            face_encoding_list.append(face_encoding)

        return face_encoding_list

    # setup_db
    def setup_database(self, img_path):
        # setup 성공한 이미지 / 전체 이미지
        totla_image = len(os.listdir(img_path))
        count = 0
        not_face = []

        # ./test_images/test1/ 내의 모든 파일
        for p in os.listdir(img_path):
            # 파일 이름을 key로
            identity = str.split(p, '.')[0]
            print("file {} embedding...".format(p))
            # 해당 이미지 파일 경로를 _img로 해놓고
            img_ = os.path.join(img_path, p)
            # 해당 이미지에대한 embedding값 구해서 encodings에 넣어줌
            encodings = self.get_face_embedding(img_)
            # 실패한 파일들 볼 수 있도록 append
            if encodings == 0:
                not_face.append(p)
                continue
            elif len(encodings) == 1:
                self.face_database[identity] = encodings[0]
            # 사진에 여러 얼굴 검출되면
            # [array1, array2] 처럼 하나의key에 각 얼굴들의 embedding 값을 넣어 줌
            else:
                self.face_database[identity] = [e for e in encodings]
            # encodings(embedding 값)은 128차원의 특징 값
            count += 1
            print("{} / {}".format(count, totla_image))
        print("얼굴을 못 읽는 이미지 : {}".format(not_face))

    # 이미지가 같은지 embedding된 값으로 비교
    # tolerance 이 더 낮으면 엄격함, 0.6을 추천함
    def compare_img(self, img_path, encoding_check_img, tolerance=0.6):
        encoding_origin_img = self.get_face_embedding(img_path)
        # original 이미지는 얼굴이 하나라 가정
        encoding_origin_img = encoding_origin_img[0]

        # 0은 얼굴이 검출되지 않았단 거임
        if encoding_origin_img == 0:
            return False

        distance_diff = []

        # 한 사진에 여러 얼굴이 검출되면 list형태로 저장해서 key에 넣어두기로 해놨었음
        if type(encoding_check_img[0]) is list:

            face_diff = []
            dist_diff = []
            # 여기서 encdoing_check_img는 이중list의 형태
            # [[np.array], [np.array]]
            for e in encoding_check_img:
                # 해당 embedding 값들과 비교하고자하는 original image 얼굴의 embedding값 비교
                # face_recognition.face_distance는 내부적으로
                # 두 벡터간의 차를 구하고 그 차에대해 벡터의 길이를 구함
                # 즉 길이가 짧을 수록(값이 작을 수록) 두 얼굴은 일치
                diff = face_recognition.face_distance(encoding_origin_img, e[0])
                # 벡터간 거리
                dist_diff.append(diff)
                print(diff)
                face_diff.append(list(diff <= tolerance))
            distance_diff.extend(min(dist_diff))

            print(face_diff)

            return distance_diff, [any(list(face_diff))]
        # 한 사진에 한 얼굴만 검출되면 np.array 타입으로 반환하기때문에 바로 비교
        face_diff = face_recognition.face_distance(encoding_origin_img, encoding_check_img[0])
        print(face_diff)
        distance_diff.append(face_diff)
        return distance_diff, list(face_diff <= tolerance)
        # compare_faces는 내부적으로
        # 128차원의 embedding된 값의 차로 얼굴간의 거리를 구한뒤
        # np.linalg.norm를 통해 128차원의 벡터의 길이를 구함
        # 이길이가 tolerance보다 작을때 두 이미지는 같은 이미지로 간주
        # tolearnace의 값을 높이면 비교연산이 좀더 관대해지게됨

    # 들어온 이미지와 DB를 비교
    def compare_db(self, img_path):
        bool_list = []
        for k in self.face_database.keys():
            _, compare_check = self.compare_img(img_path, self.face_database[k], 0.6)
            if compare_check:
                print('{}는 {}와 일치합니다'.format(os.path.split(img_path)[-1], k))
            else:
                print('{}는 {}와 불 일치합니다'.format(os.path.split(img_path)[-1], k))

            bool_list.extend(compare_check)
        return bool_list

    # 가장 유사한 이미지 return
    def similar_compare_db(self, img_path):
        diff_list = []
        db_diff = {}
        for i, k in enumerate(self.face_database.keys()):
            db_diff[i] = k
            distance_diff, compare_check = self.compare_img(img_path, self.face_database[k], 0.6)
            # print(distance_diff)
            # if compare_check:
            #     print('{}는 {}와 일치합니다'.format(os.path.split(img_path)[-1], k))
            # else:
            #     print('{}는 {}와 불 일치합니다'.format(os.path.split(img_path)[-1], k))
            diff_list.extend(distance_diff)
        min_diff = np.array(diff_list).argmin()
        print('{}가 {}만큼 가까워서 제일 유사한 얼굴'.format(db_diff[min_diff], diff_list[0]))
        return db_diff[min_diff]

# db_image_path = './test_images/test1/'
# test1 = Face_Embedding()
# test_db = test1.setup_database(db_image_path)
# print(test_db.keys())
# print(test1.compare_img('./test_images/kim1.jpg', test_db['kim2']))
# #print(test1.compare_img('./test_images/kim1.jpg', test_db['test1']))
# print(test1.compare_db('./test_images/kim1.jpg'))
# print(test1.similar_compare_db('./test_images/kim1.jpg'))

# 이 파일을 직접실행할 경우
if __name__ == '__main__':
    db_image_path = './test_images/test1/'
    test1 = Face_Embedding()
    test1.setup_database(db_image_path)
    print("*" * 100)
    print(test1.face_database.keys())
    print("*" * 100)
    print(test1.compare_db('./test_images/kim1.jpg'))
    print("*" * 100)
    print(test1.similar_compare_db('./test_images/kim1.jpg'))


