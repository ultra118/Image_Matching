from flaskapp import app
from flask import request, render_template, url_for, flash, redirect
from werkzeug import secure_filename
import sys
from flaskapp.face_rec import Image_Preprocessing,Face_Embedding
import cv2
import numpy
import json
from collections import OrderedDict
import base64
from imageio import imread
import io
from flask import jsonify
import boto3
from watson_developer_cloud import VisualRecognitionV3
from ibm_watson import VisualRecognitionV3 as VR
import glob
import ntpath
from pathlib import Path
from PIL import Image

DB_IMAGE_PATH = './flaskapp/static/images/'
embedding_model = Face_Embedding()
embedding_model.create_facedict()
embedding_model.embedding_to_csv()
embedding_model.embedding_csv_db = embedding_model.load_csv_db()


@app.route("/")
@app.route("/home")
def home():
	return render_template('home.html', title='Home')

# 이미지를 올리는 페이지 
@app.route('/upload')
def load_file():
	return render_template('upload.html', title='upload')
# 참조링크 : https://stackoverflow.com/questions/47515243/reading-image-file-file-storage-object-using-cv2
@app.route('/embedding', methods = ['GET', 'POST'])
def embedding_file():
	if request.method == 'POST':
		
		# request로 부터 파일형태를 numpy 형태로 바꿔줌 
		npimg = numpy.fromfile(request.files['file'], numpy.uint8)
		# 그리고 numpy를 image로 읽음 
		img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
		
		embedding_value = embedding_model.get_face_embedding(img, False)
		# embedding_value = embedding_model.get_face_embedding('flaskapp/static/images/kim10.jpg')
		# print(img.shape, file=sys.stdout)
		# print(img, file=sys.stdout)
		# print(embedding_value, file=sys.stdout)
		print(embedding_value, file=sys.stdout)

		# print(secure_filename(f.filename), file=sys.stdout)
		# f.save(UPLOAD_PATH + secure_filename(f.filename))
		return render_template('embedding.html', embedding=embedding_value, title='Embedding')
	return render_template('upload.html', title='upload')


# @app.route('/find', methods = ['GET', 'POST'])
# def find_same_face():
# 	if request.method == 'POST':
# 		npimg = numpy.fromfile(request.files['file'], numpy.uint8)
# 		img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
# 		embedding_value = embedding_model.get_face_embedding(img, False)
# 		embedding_model.setup_database(DB_IMAGE_PATH)
# 		# db_image_path = 'C:/TakeMeHome/IM/opencv_project/TMH_PJ/test_images/test1'
# 		# test1 = Face_Embedding()
# 		# test_img = cv2.imread('C:/TakeMeHome/IM/opencv_project/TMH_PJ/test_images/kim1.jpg')
# 		# test1.setup_database(db_image_path)
# 		# test1.similar_compare_db(test_img, path_bool=False)
# 		print(embedding_model.similar_compare_db(img, path_bool=False), file=sys.stdout)
# 		return "sucess!"
# 	return render_template('find.html', title='find')


@app.route('/compare', methods = ['GET', 'POST'])
def compare_two_images():
	if request.method == 'POST':
		npimg1 = numpy.fromfile(request.files['file1'], numpy.uint8)
		img1 = cv2.imdecode(npimg1, cv2.IMREAD_COLOR)
		npimg2 = numpy.fromfile(request.files['file2'], numpy.uint8)
		img2 = cv2.imdecode(npimg2, cv2.IMREAD_COLOR)

		embedding_value1 = embedding_model.get_face_embedding(img1, False)
		embedding_value2 = embedding_model.get_face_embedding(img2, False)
		# compare_img(self, img_path, encoding_check_img, path_bool = True, tolerance=0.6):
		# return diff_distance, matching_value
		diff_distance, matching_bool_list = embedding_model.compare_img(embedding_value1, embedding_value2, path_bool = False)
		response_data = OrderedDict()
		response_data['matching_score'] = str(diff_distance[0])
		response_data['matching_bool'] = str(matching_bool_list[0])
		
		return json.dumps(response_data, ensure_ascii = False, indent="\t")

	return render_template('compare.html', title='compare')

@app.route('/compare_db', methods = ['GET', 'POST'])
def compare_db():
	if request.method == 'POST':
		npimg = numpy.fromfile(request.files['file'], numpy.uint8)
		img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
		
		embedding_value = embedding_model.get_face_embedding(img, path_bool =False)
		results_all = {}
		if embedding_value == 0:
			results_all['facereco'] = {'NO CLASS':'NO FACE'}
			return jsonify(results_all)
		elif len(embedding_value) > 1:
			results_all['facereco'] = {'NO CLASS':'MANY FACES'}
			return jsonify(results_all)
		embedding_model.compare_csvdb(embedding_value, path_bool=False)
		
		return "sucess!"

	return render_template('compare_db.html', title='compare_db')

# set FLASK_APP=flaskapp.py
# flask run --host=0.0.0.0
@app.route('/similar_db', methods = ['GET', 'POST'])
def similar_db():
	if request.method == 'POST':
		npimg = numpy.fromfile(request.files['file'], numpy.uint8)
		img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
		
		embedding_value = embedding_model.get_face_embedding(img, path_bool =False)
		results_all = {}
		if embedding_value == 0:
			results_all['facereco'] = {'NO MATCH':'NO FACE'}
			return jsonify(results_all)
		elif len(embedding_value) > 1:
			results_all['facereco'] = {'NO MATCH':'MANY FACES'}
			return jsonify(results_all)
		matching_file, matching_distance = embedding_model.similar_compare_csvdb(embedding_value, path_bool=False)
		if matching_distance[0] > 0.4:
			matching_file = 'NO MATCH'
			matching_distance[0] = 0
		else:
			matching_distance[0] = (1 - matching_distance[0]*1.25) * 100
		results_all['facereco'] = {matching_file : "{:3.2f}%".format(matching_distance[0])}
		return jsonify(results_all)

	return render_template('similar_db.html', title='similar_db')

# Face recognition REST API
@app.route("/find2/<string:auth>", methods=['GET', 'POST'])
def find_them2(auth):
	if auth and auth=="them":
		data = request.get_data(cache=True, as_text=True, parse_form_data=False)
		image_src = base64.b64decode(data)
		filename = './flaskapp/static/cache_images/some_image.jpg'
		with open(filename, 'wb') as f:
			f.write(image_src)

		embedding_value = embedding_model.get_face_embedding(filename, path_bool=True, one_image=True)
		results_all = {}
		print("embedding value : {}".format(embedding_value))
		print("embedding value len : {}".format(len(embedding_value)))
		if embedding_value == 0:
			results_all['facereco'] = {'NO MATCH':'NO FACE'}
			return jsonify(results_all)
		elif len(embedding_value) > 1:
			results_all['facereco'] = {'NO MATCH':'MANY FACES'}
			return jsonify(results_all)
		matching_file, matching_distance = embedding_model.similar_compare_csvdb(embedding_value, path_bool=False)
		if matching_distance[0] > 0.4:
			matching_file = 'NO MATCH'
			matching_distance[0] = 0
		else:
			matching_distance[0] = (1 - matching_distance[0]*1.25) * 100
		results_all['facereco'] = {matching_file : "{:3.2f}%".format(matching_distance[0])}
		return jsonify(results_all)
	return "failed"

# RESTAPI => if request POST, return JSON
@app.route("/find/<string:auth>", methods=['GET', 'POST'])
def find_them(auth):
	if auth and auth=="them":
		data = request.get_data(cache=True, as_text=True, parse_form_data=False)
		image_src = base64.b64decode(data)
		filename = './flaskapp/static/cache_images/some_image.jpg'
		with open(filename, 'wb') as f:
			f.write(image_src)

    # WATSON visual recognition SECTION
		visual_recognition = VR(
			'2018-03-19',
			iam_apikey='SxPQAFpzbGJU3pYhPazzWdZiGoo7gM7WLSUN2F4FWRcb')

		with open(filename,'rb') as images_file:
			classes = visual_recognition.classify(
				images_file,
				threshold='0.8',
				classifier_ids=["face_match_957902511"]).get_result()

		try:
			watson_class = classes['images'][0]['classifiers'][0]['classes'][0]['class']
			watson_score = float(classes['images'][0]['classifiers'][0]['classes'][0]['score']*100)
			watson_result = ["{:3.2f}%".format(watson_score), watson_class]
		# watson_result = "{:3.2f}% match with {}".format(watson_score, watson_class)
		except Exception as e:
			watson_class = 'NO MATCH'
			watson_score = '0'
			watson_result = [watson_score, watson_class]

    # THIS HERE STARTS AMAZONE SERVICE
		folder_dir = "./flaskapp/static/dataset"

		image_files = []

		jpg_files = Path(folder_dir).glob('*.jpg')
		png_files = Path(folder_dir).glob('*.png')
	# to read subfoldersas well,
	# png_files = Path(folder_dir).glob('**/*.png')

		for f in jpg_files:
			file_path = str(f)
			image_files.append(file_path)

		for f in png_files:
			file_path = str(f)
			image_files.append(file_path)

		print(image_files)

		amwResultArr = []

		for path in image_files:
			targetSRC = str(path)
			targetOPEN = Image.open(targetSRC, mode='r')
			targetByteArr = io.BytesIO()
			targetOPEN.save(targetByteArr, format='PNG')
			targetByteArr = targetByteArr.getvalue()
			targetOPEN.close()

      		# AMW face rekognition SECTION
			client=boto3.client('rekognition')
			try:
				imageSource = open(filename,'rb')
				amw = client.compare_faces(SimilarityThreshold=80,
					SourceImage={'Bytes': imageSource.read()},
					TargetImage={'Bytes': targetByteArr})
				imageSource.close()
			except Exception as e:
				amwResultArr.append('Failure to load target image')
				print('Failure to load target image')
			else:
				print('check ================================================')
				print('{} is being processed'.format(targetSRC))

				# IF there's a match,
				if len(amw['FaceMatches']) == 1:
					# match percentage is float
					amw_similarity = amw['FaceMatches'][0]['Similarity']
					# format the float
					amw_result = "{:3.2f}%".format(amw_similarity)
					amwResultArr.append(amw_result)
				else:
					amw_result = "No Match"
					amwResultArr.append(amw_result)

		dataset_names = ['Emma Watson','Harry Potter','Jaden Smith','Matilda','Finn']
		amazon_result = ["0", "No Match Found"]
		result_idx = 400
		for r in amwResultArr:
			if r != 'No Match':
				result_idx = dataset_names[amwResultArr.index(r)]
				amazon_result = [r, result_idx]

		results_all = {}
		results_all['watson'] = watson_result
		results_all['amazon'] = amazon_result

		# THIS HERE STARTS FACE_RECOGNITION SERVICE

		embedding_value = embedding_model.get_face_embedding(filename, path_bool=True, one_image=True)
		if embedding_value == 0:
			results_all['facereco'] = [[0, '얼굴이 검출되지 않음']]	
			return jsonify(results_all)
		elif len(embedding_value) > 1:
			results_all['facereco'] = [[1, '얼굴이 너무 많이 검출 됨']]	
			return jsonify(results_all)

		matching_file, matching_distance = embedding_model.similar_compare_csvdb(embedding_value, path_bool=False)
		
		results_all['facereco'] = [[matching_file, matching_distance[0]]]
		return jsonify(results_all)

	return "API CALL FAILURE"