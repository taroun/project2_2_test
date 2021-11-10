import datetime

# Create your views here.
from django.db import transaction
from django.forms import modelformset_factory
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.template import RequestContext
from django.urls import reverse
from django.views.generic import CreateView, DetailView
from easyocr import easyocr

from animateapp.forms import AnimateForm
from animateapp.models import Animate, AnimateImage

import sys
from PIL import Image
import cv2.cv2 as cv2
import numpy as np

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
            full_list = []
            for f in files:
                l_r = str(form.instance.left_right)
                image = f
                animateimage = AnimateImage(animate=animate, image=f)
                animateimage.save()
                image_len_list = crop(image, l_r)
                full_list.extend(image_len_list)

            video_path = view_seconds(full_list)
            # video_list = view_seconds(full_list)
            # video_path = effect_video(video_list)
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


#임시로 이미지 분할
#앞으로 컷분리 모델이 들어갈 부분
def crop(f, l_r):
    #이미지 저장위치로 이미지 받아옮
    im = Image.open(f)
    #이미지 width, height 값 받아옮
    img_width, img_height = im.size

    #opencv 이미지 읽어오기
    numpy_image = np.array(im)
    src = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    #src = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    #이미지 3등분하기위한 높이값 나눔
    hei_3 = int(img_height / 3)
    #잘려진 cut과 순서 글자수를 넣을 리스트 미리생성
    crop_img = []
    # 이미지를 자른다.
    for i in range(3):
        #이미지를 [높이 나누는 부분: 넓이 전체]로 복사함
        dst = src[hei_3*i:hei_3*(i+1):, 0:img_width].copy()
        #img_text_easyocr(dst)-잘려진cut으로 이미지로 글자수 측정 함수
        #반환값은 cut의 글자수
        txt_len = img_text_easyocr(dst)
        #리스트에 순서대로 잘라서 cut image, 글자수 순으로 추가
        crop_img.append([dst, txt_len])
    #cut의 [순서,이미지,글자수]의 리스트 반환
    return crop_img


#인식률이 좋은 easyocr버전 이미지 받아 글자수 반환해주는 함수
def img_text_easyocr(img):
    #인식 언어 설정
    reader = easyocr.Reader(['ko', 'en'])
    #이미지를 받아 문자열 리스트를 반환해줌
    result = reader.readtext(img, detail=0)
    #리스트 원소 합쳐서 문자여 총 길이 확인
    text_result = " ".join(result)
    text_result_len = len(text_result)
    print("길이:" + str(len(text_result)))
    print(text_result)
    #문자열 길이 반환
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
    fourcc = cv2.VideoWriter_fourcc(*'H264')

    # 현재 영상 프레임을 첫번째이미지로 설정(변경가능)
    frame = image_list[0][0]
    height, width, layers = frame.shape

    # video작성부분(저장위치, codec, fps, 영상프레임)
    video = cv2.VideoWriter(out_path, fourcc, 10.0, (width, height))
    # 리스트에서 한 cut씩 가져옮
    for image in image_list:
        # 기본 5초에 이미지의 글자수를 10으로 나눈만큼 반복하여 같은 이미지 기록
        each_image_duration = 3*10 + int(image[1])
        for _ in range(each_image_duration):
            video.write(image[0])

    # 객체를 반드시 종료시켜주어야 한다
    video.release()
    # 모든 화면 종료해준다.
    #cv2.destroyAllWindows()

    # 영상 저장 위치 반환
    return video_name

# #[이미지,글자수]의 리스트를 받아 영상으로 만들고 저장하는 함수
# def view_seconds(image_list):
#     #video codec 설정
#     fourcc = cv2.VideoWriter_fourcc(*'H264')
#
#     #현재 영상 프레임을 첫번째이미지로 설정(변경가능)
#     frame = image_list[0][0]
#     height, width, layers = frame.shape
#
#     video_list = []
#     #리스트에서 한 cut씩 가져옮
#     for i, image in enumerate(image_list):
#         # 영상 저장 위치 설정
#         video_name = 'ani/ani_cut/' + str(i) + '.mp4'
#         out_path = 'media/' + video_name
#         # video작성부분(저장위치, codec, fps, 영상프레임)
#         video = cv2.VideoWriter(out_path, fourcc, 10.0, (width, height))
#         #기본 5초에 이미지의 글자수를 10으로 나눈만큼 반복하여 같은 이미지 기록
#         each_image_duration = 4*10 + int(image[1])
#         for _ in range(each_image_duration):
#             video.write(image[0])
#         video.release()
#         video_list.append(str(out_path))
#
#     return video_list

#
# def effect_video(video_list):
#     print(video_list)
#
#     video_name = 'ani/animate.mp4'
#     out_path = 'media/' + video_name
#
#     for i in range(1, len(video_list)):
#         if i <= 1:
#             print(i)
#             cap1 = cv2.VideoCapture(video_list[i-1])
#         else:
#             print(out_path)
#             cap1 = cv2.VideoCapture(out_path)
#         cap2 = cv2.VideoCapture(video_list[i])
#
#         if not cap1.isOpened() or not cap2.isOpened():
#             print('video open failed!')
#             sys.exit()
#
#         print(cap1)
#         print(cap2)
#         # 두 동영상의 크기, FPS는 같다고 가정
#         frame_cnt1 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_cnt2 = round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap1.get(cv2.CAP_PROP_FPS)
#         effect_frames = int(fps * 1)  # 전환 속도를 결정
#
#         print('frame_cnt1:', frame_cnt1)
#         print('frame_cnt2:', frame_cnt2)
#         print('FPS:', fps)
#
#         #delay = int(1000 / fps)
#
#         w = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fourcc = cv2.VideoWriter_fourcc(*'H264')
#
#         # 출력 동영상 객체 생성
#         out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
#
#         # 1번 동영상
#         for y in range(frame_cnt1 - effect_frames):
#             ret1, frame1 = cap1.read()
#
#             if not ret1:
#                 break
#
#             out.write(frame1)
#             #cv2.imshow('frame', frame1)
#             #cv2.waitKey(delay)
#
#         # 합성 과정
#         for k in range(effect_frames):
#             ret1, frame1 = cap1.read()
#             ret2, frame2 = cap2.read()
#
#             if not ret1 or not ret2:
#                 print(ret1)
#                 print(ret2)
#                 print('frame read error!')
#                 sys.exit()
#
#             dx = int(w / effect_frames * k)
#             # frame_new = np.zeros((h, w, 3), dtype=np.uint8)
#             # frame_new[:, 0:dx, :] = frame2[:, 0:dx, :]  # 2번 동영상 먼저 등장
#             # frame_new[:, dx:w, :] = frame1[:, dx:w, :]  # 1번 동영상 점차 사라짐
#
#             # 디졸브 효과
#             # 과중치를 이용. cv2.addWeighted 함수 이용하면 된다.
#             alpha = i / effect_frames
#             frame_new = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
#
#             out.write(frame_new)
#             #cv2.imshow('frame', frame_new)
#             #cv2.waitKey(delay)
#
#         # 2번 동영상
#         for j in range(effect_frames, frame_cnt2):
#             ret2, frame2 = cap2.read()
#
#             if not ret2:
#                 break
#
#             out.write(frame2)
#             #cv2.imshow('frame', frame2)
#             #cv2.waitKey(delay)
#
#         cap1.release()
#         cap2.release()
#         out.release()
#
#         os.remove(video_list[i])
#     os.remove(video_list[0])
#
#     return video_name
