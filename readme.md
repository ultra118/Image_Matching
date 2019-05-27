# Take Me Home Project

## image matching service model

### AWS face rekognition API를 활용해 얼굴 매칭

### face_recognition pgk를 활용해 얼굴 매칭

- 각 얼굴에 대해 128개의 특징값을 뽑고 이 값들로 얼굴 비교 가능  
- 링크 : https://ukayzm.github.io/python-face-recognition/

#### 필요 package 설치(windows)

```bash
C:> pip install opencv-python
```

```bash
C:> pip install opencv-contrib-python
```

```bash
C:> pip install dlib
```

- dlib-19.8.1-cp36-cp36m-win_amd64.whl를 커맨드라인의 해당 경로로 옮겨놓은 후

```bash
C:> pip install dlib-19.8.1-cp36-cp36m-win_amd64.wh
```

```bash
C:> pip install face_recognition
```



### Flask APP Test

```bash
C:> pip instal flask
```

- 서로 다른 두 사진의 얼굴을 비교하는 서비스
- 하나의 이미지를 업로드해 .csv로 저장되어있는 embedding값과 비교해 매칭률을 보여주는 서비스
- 하나의 이미지를 업로드해 .csv로 저장되어있는 embedding파일과 비교해 가장 유사한 이미지 리턴



### 간소화된 one-shot model를 활용한 얼굴 매칭

- [(batch, 200, 200, 3), (batch, 200, 200, 3)] 의 형태로 input을 받고 각각 대칭되는 CNN layer를 통해 특징 값을 뽑고 flatten 해서 그 차를 구해내고 dense layer에 input으로 넣어줌

