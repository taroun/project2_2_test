import datetime

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
            full_list = []
            for f in files:
                l_r = str(form.instance.left_right)
                animate_image = AnimateImage(animate=animate, image=f)
                animate_image.save()
                im = Image.open(f)
                img_array = np.array(im)
                image_list.append(img_array)

            cuts = make_cut(image_list)
            print(len(cuts))
            image_len_list = image_len(cuts)
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

    width = 410
    height = 500
    back_image = np.ones((width, height, 3), np.uint8)*255
    cols, rows, channel = cut.shape
    space_width = int((width - cols) / 2)
    space_height = int((height - rows) / 2)
    back_image[space_width:space_width + cols, space_height:space_height + rows] = cut

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
    # image = Image.fromarray(np.uint8(cm.plasma(img) * 255))
    # img_resize = image.resize((int(image.width / 2), int(image.height / 2)))
    #인식 언어 설정
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
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
