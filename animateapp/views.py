import datetime

import torch.cuda
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views.generic import CreateView, DetailView
from easyocr import easyocr
from matplotlib import cm

from animateapp.forms import AnimateForm
from animateapp.models import Animate, AnimateImage

from PIL import Image
import cv2.cv2 as cv2
from tensorflow.python.keras.models import load_model
import numpy as np
import base64
import gc

#충돌 오류 때문에 기록...
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class AnimateCreateView(CreateView):
    model = Animate
    form_class = AnimateForm
    template_name = 'animateapp/create.html'

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)

        files = request.FILES.getlist('image')
        form.instance.image = files[0]
        animate = form.save(commit=False)
        animate.save()
        if form.is_valid():
            image_list = []
            for f in files:
                l_r = str(form.instance.left_right)
                animate_image = AnimateImage(animate=animate, image=f, image_base64="")
                animate_image.save()
                path = 'media/' + str(animate_image.image)
                with open(path, 'rb') as img:
                    base64_string = base64.b64encode(img.read())
                    tmp = str(base64_string)[2:-1]
                animate_image.image_base64 = tmp
                animate_image.save()
                im = Image.open(f)
                img_array = np.array(im)
                image_list.append(img_array)

            cuts = make_cut(image_list)
            image_len_list = image_len(cuts)

            video_path = view_seconds(image_len_list)
            animate = form.save(commit=False)
            animate.ani = video_path
            animate.save()

            return HttpResponseRedirect(reverse('animateapp:detail', kwargs={'pk': animate.pk}))
        else:
            return self.form_invalid(animate)


class AnimateDetailView(DetailView):
    model = Animate
    context_object_name = 'target_animate'
    template_name = 'animateapp/detail.html'


# 이미지 별 사이즈 적용 수정 필요
# 컷 분리 함수
def split_cut(img, polygon):
    x, y, w, h = cv2.boundingRect(polygon)
    croped = img[y:y + h, x:x + w].copy()
    pts = polygon - polygon.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    bg = np.ones_like(croped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    cut = bg + dst

    height = 650
    width = 450
    back_image = np.ones((height, width, 3), np.uint8)*255
    cols, rows, channel = cut.shape
    space_height = int((height - cols) / 2)
    space_width = int((width - rows) / 2)
    back_image[space_height:space_height + cols, space_width:space_width + rows] = cut

    return back_image


# 컷 정렬 함수
def sort_cut(contours):
    n = 0
    centroids = []
    for contour in contours:
        centroid = cv2.moments(contour)
        cx = int(centroid['m10']/centroid['m00'])
        cy = int(centroid['m01']/centroid['m00'])
        centroids.append([cx, cy, n])
        n += 1
    centroids.sort(key=lambda x: (x[1], x[0]))
    centroids = np.array(centroids)
    index = centroids[:, 2].tolist()
    sort_contours = [contours[i] for i in index]
    return sort_contours


def make_cut(img_list):
    IMAGE_SIZE = 224
    model = load_model('model/best_gray_model_2.h5')
    img_input = []
    cuts = []
    # 모델 입력 전처리
    img_gray = cv2.cvtColor(img_list[0], cv2.COLOR_BGR2GRAY)
    for img in img_list:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_res = cv2.resize(img_gray, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        img_input.append(img_gray_res/255)
    img_input = np.asarray(img_input)
    img_input = img_input.reshape(img_input.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
    #모델 적용
    img_predict = model.predict(img_input).reshape(len(img_input), IMAGE_SIZE, IMAGE_SIZE)
    #출력 이미지 전처리
    labels = [np.around(label)*255 for label in img_predict]
    labels = [cv2.resize(label, dsize=img_gray.shape[::-1], interpolation=cv2.INTER_AREA) for label in labels]
    for idx, label in enumerate(labels):
        label = np.asarray(label, dtype=np.uint8)
        contours, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 컷 정렬
        contours = sort_cut(contours)
        background = np.full(label.shape, 255, dtype=np.uint8)
        polygons = [contour.reshape(contour.shape[0], 2) for contour in contours]
        for polygon in polygons:
            x, y, w, h = cv2.boundingRect(polygon)
            if w > 50:
                cuts.append(split_cut(img_list[idx], polygon))
    return cuts


#이미지 길이처리
def image_len(cut_list):
    cut_img_list = []
    # 이미지를 자른다.
    for cut in cut_list:
        txt_len = img_text_easyocr(cut)
        #리스트에 순서대로 잘라서 cut image, 글자수 순으로 추가
        cut_img_list.append([cut, txt_len])
    #cut의 [이미지,글자수]의 리스트 반환
    return cut_img_list


#인식률이 좋은 easyocr버전 이미지 받아 글자수 반환해주는 함수
def img_text_easyocr(img):
    gc.collect()
    torch.cuda.empty_cache()
    print(img.shape)
    img_result = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
    #인식 언어 설정
    reader = easyocr.Reader(['ko', 'en'])
    #이미지를 받아 문자열 리스트를 반환해줌
    result = reader.readtext(img_result, detail=0)
    #리스트 원소 합쳐서 문자여 총 길이 확인
    text_result = " ".join(result)
    text_result_len = len(text_result)
    #문자열 길이 반환
    print("길이"+str(text_result_len))
    print(text_result)
    return text_result_len


#[이미지,글자수]의 리스트를 받아 영상으로 만들고 저장하는 함수
def view_seconds(image_list):
    # 영상이름 오늘 날자와 시간으로 지정
    nowdate = datetime.datetime.now()
    daytime = nowdate.strftime("%Y-%m-%d_%H%M%S")
    # 영상 저장 위치 설정
    video_name = 'ani/' + daytime + '.mp4'
    out_path = 'media/' + video_name
    # video codec 설정
    fourcc = cv2.VideoWriter_fourcc(*'AVC1')

    # 현재 영상 프레임을 첫번째이미지로 설정(변경가능)
    frame = image_list[0][0]
    height, width, layers = frame.shape

    # video작성부분(저장위치, codec, fps, 영상프레임)
    fps = 24.0
    video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    # 리스트에서 한 cut씩 가져옮
    for i, image in enumerate(image_list):
        # wid, hei = 1280, 720
        # image_hei, image_wid = image.shape[:2]
        # img_result = cv2.resize(cut, (1100, (1100/image_wid) * image_hei), interpolation = cv2.INTER_CUBIC)
        # sr = cv2.dnn_superres.DnnSuperResImpl_create()
        # sr.readModel('models/EDSR_x3.pb')
        # sr.setModel('edsr', 3)
        # result = sr.upsample(img_result)

        # back_image = np.ones((hei, wid, 3), np.uint8) * 255
        # cols, rows, channel = result.shape
        # space_width = int((wid - rows) / 2)

        # 기본 5초에 이미지의 글자수를 10으로 나눈만큼 반복하여 같은 이미지 기록
        each_image_duration = int(2*fps) + int(image[1]*(fps/10))

        for k in range(each_image_duration):
        #   if cols < hei:
        #       nx = int((hei-720 / each_image_duration) * k)

        #       back_image[:, space_width:space_width + rows] = result[nx:nx+720,:]
        #       video.write(back_image)
        #   else:
        #       space_height = int((height - cols) / 2)
        #       back_image[space_height:space_height + cols, space_width:space_width + rows] = result
        #       video.write(back_image)

            video.write(image[0])

        #영상 전환 효과
        if i+1 < len(image_list):
            # #잘리면서 그림 바뀜
            # for j in range(1, int(fps+1)):
            #     dx = int((width / fps) * j)
            #     frame = np.zeros((height, width, 3), dtype=np.uint8)
            #     frame[:, 0:dx, :] = image[0][:, width-dx:width, :]
            #     frame[:, dx:width, :] = image_list[i+1][0][:, 0:width-dx, :]
            #     video.write(frame)
            #디졸브 효과
            for j in range(1, int(fps+1)):
                alpha = j / fps
                frame = cv2.addWeighted(image[0], 1 - alpha, image_list[i+1][0], alpha, 0)
                video.write(frame)

    # 객체를 반드시 종료시켜주어야 한다
    video.release()

    # 영상 저장 위치 반환
    return video_name
