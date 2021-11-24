import datetime

from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views.generic import CreateView, DetailView

from animateapp.forms import AnimateForm
from animateapp.models import Animate, AnimateImage

from PIL import Image
import cv2.cv2 as cv2
from tensorflow.python.keras.models import load_model
import numpy as np
import base64
import gc
import tensorflow as tf
from sklearn import preprocessing


# 충돌 오류 때문에 기록...
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

IMAGE_SIZE = 224

# 컷 모델
model = load_model('model/best_gray_model_2.h5')
# 말풍선 모델
model_b = load_model('model/bubble_gray_model.h5')


class AnimateCreateView(CreateView):
    model = Animate
    form_class = AnimateForm
    template_name = 'animateapp/create.html'

    def post(self, request, *args, **kwargs):
        # form 받아옴
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        # 이미지 파일들만 받아오기
        files = request.FILES.getlist('image')
        # Animate 모델에는 첫이미지만 저장되게 하기
        form.instance.image = files[0]
        animate = form.save(commit=False)
        animate.save()

        if form.is_valid():
            # 이미지 파일 np array 형식으로 저장하기위한 리스트 생성
            image_list = []
            # 모델 돌리고 정리된 이미지들 들어오는 부분
            img_list = []
            # 선택값들 받아오기
            l_r = str(form.instance.left_right)
            t_c = str(form.instance.toon_comic)
            ani_effect = str(form.instance.ani_effect)
            tran_effect = str(form.instance.transition_effect)

            # 이미지 파일을 하나씩 불러와서 np array 형식으로 image_list 에 저장
            for f in files:
                animate_image = AnimateImage(animate=animate, image=f, image_base64="", left_right=l_r, toon_comic=t_c,
                                             ani_effect=ani_effect, transition_effect=tran_effect)
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

            # toon 방식
            if t_c == 'T':
                # 전처리
                labels_cut, labels_bubble = image_preproc(image_list)
                # 컷 분리
                bubbles, centroids, bubble_centers = make_cut_bubble(image_list, labels_bubble, l_r, is_bubble=True)
                cuts, centroids_cut, polygons = make_cut_bubble(image_list, labels_cut, l_r, is_bubble=False)
                # 객체 생성과 동시에 말풍선과 컷을 매칭합니다.
                FrameBook = ComicFrameBook(ani_effect, bubbles, cuts, polygons, bubble_centers,
                                           page_len=len(image_list))
                text_bubble_len_list = FrameBook.txt_per_bub_list
                img_list = FrameBook.makeframe_proc()

            # =========================================== 수정된 부분 =============================================
            # comic 책 방식
            else:
                img_list, text_bubble_len_list = make_page_cut(image_list, l_r)
            # ====================================================================================================

            # 반환값은 저장된 영상위치
            video_path = view_seconds(img_list, t_c, ani_effect, tran_effect, text_bubble_len_list)

            # gpu session 비워주기
            tf.keras.backend.clear_session()
            gc.collect()
            # 다시 model에 저장
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


# 말풍선 모델
def image_preproc(img_list):  # 이 코드는 전처리부분만을 가져왔습니다.
    img_input = []

    # 모델 입력 전처리
    for img in img_list:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_res = cv2.resize(img_gray, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        img_input.append(img_gray_res / 255)
    img_input = np.asarray(img_input)
    img_input = img_input.reshape(img_input.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)

    # 모델 적용 컷 or 말풍선
    cut_predict = model.predict(img_input).reshape(len(img_input), IMAGE_SIZE, IMAGE_SIZE)
    bubble_predict = model_b.predict(img_input).reshape(len(img_input), IMAGE_SIZE, IMAGE_SIZE)

    # 출력 이미지 전처리
    # 이제 컷과 말풍선을 동시에 처리합니다.
    labels_list = []
    for predict in [cut_predict, bubble_predict]:
        labels = [np.around(label) * 255 for label in predict]
        # 여기서 다시 출력이미지 사이즈를 키움
        labels = [cv2.resize(label, dsize=img_gray.shape[::-1], interpolation=cv2.INTER_AREA) for label in labels]
        labels_list.append(labels)

    return labels_list[0], labels_list[1]  # 0 : cut, 1 : bubble


def split_cut_b(img, polygon, page_num, is_bubble=False):
    x, y, w, h = cv2.boundingRect(polygon)  # 폴리곤으로 bounding 박스 그림
    croped = img[y:y + h, x:x + w].copy()  # 원본 이미지에서 자름.

    # 패딩을 추가해보자
    # 컷일 경우에만 패딩을 추가합니다.
    if not is_bubble:
        # ******* 현 패딩 > 컷의 높이와 너비 중 큰 값의 0.45
        # ******* 대부분의 컷은 0.3정도로도 잘 작동하는데 어떤 컷은 0.4에도 터집니다.
        # ******* ex) 2부 4화 94p
        padding = int(max(h, w) * 0.45)
        startline = int(padding / 2)

        img_padding = np.zeros([h + padding, w + padding, 3], np.uint8)
        img_padding[startline:startline + h, startline: startline + w] = croped

        croped = img_padding

        pts = polygon - polygon.min(axis=0) + startline
    else:
        pts = polygon - polygon.min(axis=0)

    mask = np.zeros([croped.shape[0], croped.shape[1]], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(croped, croped, mask=mask)  # 빈 공간에 컷만 추가.

    # 변경점. 이제 모든 이미지는 x, y, w, h, mask, matching_num 값을 포함합니다.
    # 컷에 말풍선이 없거나 말풍선이 잘못 검출 된 경우를 위해 여기서 초기화합니다.
    if is_bubble:
        txt_per_bub = find_txt_cnt(dst, mask)
        return {'image': dst, 'xywh': [x, y, w, h], 'mask': mask, 'matching_cut_num': -1, 'txt_per_bub': txt_per_bub}
    else:
        return {'image': dst, 'xywh': [x, y, w, h], 'mask': mask, 'matching_bub_num': [], 'padding': padding}


def find_txt_cnt(bubble, mask):
    _, thresh_mask = cv2.threshold(mask, 127, 1, 0) # 이진화
    bubble_px_cnt = np.sum(thresh_mask)
    bg = np.ones_like(mask)
    res = bubble.copy()
    cv2.bitwise_not(res, res, mask=bg)
    res_gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY) # 이미지 1채널로 변경
    _, res_bit = cv2.threshold(res_gray, 127, 1, 0)  ##  이진화
    only_text = cv2.bitwise_and(thresh_mask, thresh_mask, mask=res_bit)
    bubble_txt_cnt = np.sum(only_text)
    txt_per_bub = bubble_txt_cnt / bubble_px_cnt
    return txt_per_bub


def sort_cut_b(img_list, contours, read, is_bubble=False):
    height = img_list[0].shape[0]
    width = img_list[0].shape[1]
    n = 0
    centroids = []
    bubble_centers = []

    for contour in contours:
        cx = contour.min(axis=0)[0][0] // int(width / 5)  # 이것은 순서를 정하기 위해서 뽑습니다.
        cy = contour.min(axis=0)[0][1] // int(height / 5)

        centroids.append([cx, cy, n])
        n += 1

    if read == 'L':
        centroids.sort(key=lambda x: (x[1], x[0]))
    elif read == 'R':
        centroids.sort(key=lambda x: (x[1], -x[0]))
    centroids = np.array(centroids)

    index = centroids[:, 2].tolist()
    sort_contours = [contours[i] for i in index]

    if is_bubble:
        for contour in sort_contours:
            centroid = cv2.moments(contour)
            bx = int(centroid['m10'] / centroid['m00'])  # 이건 말풍선과 컷을 매칭하기 위해서 뽑습니다.
            by = int(centroid['m01'] / centroid['m00'])

            bubble_centers.append([bx, by, n])

        return sort_contours, centroids, bubble_centers  # 이걸 이용해 컷과 매칭합니다.

    return sort_contours, centroids


def make_cut_bubble(img_list, labels, read, is_bubble=False):
    cuts_list = []
    centroids_list = []
    polygons_list = []
    bubble_centers_list = []

    for idx, label in enumerate(labels):
        cuts = []
        label = np.asarray(label, dtype=np.uint8)

        contours, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # countour를 찾음.

        # 컷에 말풍선이 하나도 없을 때의 탈출조건
        if is_bubble and (len(contours) == 0):
            cuts_list.append([])
            bubble_centers_list.append([])
            continue

        # 아직 잘못 나온 이미지 처리 안함..
        # print("여기")
        # print(np.shape(contours))
        # print(contours[0].shape)
        # for cont in contours:
        #     if cont[0] < 200:
        #         contours.remove(cont)

        # 컷 정렬
        if is_bubble:
            contours, centroids, bubble_centers = sort_cut_b(img_list, contours, read, is_bubble=True)
        else:
            contours, centroids = sort_cut_b(img_list, contours, read)

        polygons = [contour.reshape(contour.shape[0], 2) for contour in contours]  # contour reshape

        if is_bubble:
            for polygon in polygons:
                cuts.append(split_cut_b(img_list[idx], polygon, page_num=idx, is_bubble=True))

            bubble_centers_list.append(bubble_centers)
        else:
            for polygon in polygons:
                cuts.append(split_cut_b(img_list[idx], polygon, page_num=idx))

            polygons_list.append(polygons)

        cuts_list.append(cuts)
        centroids_list.append(centroids)

    # 변경점. 이제 모든 출력값은 페이지마다로 구분됩니다.
    # [0] : 페이지0번, [0][0] : 페이지0번의 0번 째 컷 or 버블의 데이터
    if is_bubble:
        return cuts_list, centroids_list, bubble_centers_list  # bubble_centers 는 말풍선 컷 매칭을 위해 뽑습니다.
    else:
        return cuts_list, centroids_list, polygons_list  # 폴리곤은 말풍선 위치결정을 위해서 뽑습니다.


class ComicFrameBook():

    def __init__(self, ani_effect, bubbles, cuts, cut_polygons, bubble_centers, page_len):
        self.ani_effect = ani_effect
        self.cuts = cuts
        self.bubbles = bubbles
        self.cut_polygons = cut_polygons
        self.bubble_centers = bubble_centers

        self.page_len = page_len
        self.txt_per_bub_list = []
        self.page_count = 0  # 결과 출력용도

        for i in range(self.page_len):
            # 비어있는 풍선의 경우
            if len(self.bubbles[i]) == 0:
                continue

            self.matching_bubble2cut(self.cuts[i], self.bubbles[i], self.bubble_centers[i], self.cut_polygons[i])

    # 이건 단일 페이지에 작동합니다.
    def matching_bubble2cut(self, cuts, bubbles, bubble_centers, polygons, ):
        for i, bubble_center in enumerate(bubble_centers):
            for j, polygon in enumerate(polygons):

                # 말풍선의 중심점이 각 컷의 폴리곤 내부에 있는지 확인합니다.
                is_inpolygon = cv2.pointPolygonTest(polygon, bubble_center[:2], True)
                if is_inpolygon >= 0:
                    bubbles[i]['matching_cut_num'] = j
                    cuts[j]['matching_bub_num'].append(i)  # 클래스를 여러 번 선언하면 이 리스트가 중복되는 이슈가 있습니다.
                    break  # matching_bub_num  은 한 컷에 여러 풍선이 매칭되는 경우를 위해 리스트로 되어있습니다.
                    # 그런데 해당 리스트가 초기화 되는 부분은 컷 분리 함수이고 클래스와 동시에 동작하지 않습니다.
                    # 때문에 깊은복사 이슈가 생겨서 리스트가 중복됩니다. 아직 수정하지 못했습니다.
                    # 그래서 현재는 클래스를 다시 선언하시려면 컷 분리 부분도 다시 실행해주셔야합니다.

    # 매칭된 컷-버블 정보를 이용해 make_bubblescope_cut 함수를 실행합니다.
    def makeframe_proc(self):
        frame_pages = []
        self.page_count = 0
        for page_num in range(self.page_len):  # 페이지단위로 움직입니다.
            frames = []
            txt_per_bub_page = []
            for idx, cut in enumerate(self.cuts[page_num]):
                bub_nums = cut['matching_bub_num']  # 각 컷에 매칭되는 말풍선을 찾습니다.

                ############################
                # 말풍선이 없는 경우. 지금은 이미지 하나만 그대로 들어갑니다. 나중에 개수를 수정할 필요가 있습니다.
                if len(bub_nums) == 0:
                    frames.append([cut['image']])
                    txt_per_bub_page.append(0.05)
                    self.page_count += 1
                    continue

                # 말풍선 확대와 복사 작업을 실행합니다.
                for bub_num in bub_nums:
                    target_bub = self.bubbles[page_num][bub_num]
                    bub_centroid = self.bubble_centers[page_num][bub_num]
                    txt_per_bub_page.append(target_bub['txt_per_bub'])
                    frames.append(
                        self.make_bubblescope_cut(self.ani_effect, target_bub, cut, bub_centroid))

            frame_pages.append(frames)
            self.txt_per_bub_list.append(txt_per_bub_page)
        return frame_pages

    # 컷 한장, 버블 한장당 작동합니다.
    # 확대 배율은 현재 임의로 설정 해 두었습니다.
    def make_bubblescope_cut(self, ani_effect, bubble, cut, bub_centroid):
        padding = cut['padding']
        half_padding = int(padding / 2)
        if ani_effect == "B":
            scope_list = np.linspace(1, 1.6, 12)
        else:
            scope_list = np.linspace(1, 1, 12)

        cut_bg = cut['image']
        bub_fg = bubble['image']
        cut_xywh = cut['xywh']
        bub_xywh = bubble['xywh']

        # mask 와 mask_inv 를 만들자
        mask = bubble['mask']

        bubble_scope_cuts = []

        for scope in scope_list:
            # 강조를 위해서 resize 합니다.
            # 리사이징 때문에 mask_inv 는 여기서 만듭니다.
            resize_scale = [int(bub_xywh[2] * scope), int(bub_xywh[3] * scope)]

            bub_fg_resize = cv2.resize(bub_fg, dsize=resize_scale, interpolation=cv2.INTER_AREA)

            mask_resize = cv2.resize(mask, dsize=resize_scale, interpolation=cv2.INTER_AREA)
            mask_inv = cv2.bitwise_not(mask_resize)

            # 이미지 변경할 시작점을 잡아줍니다.
            bubble_start_h = bub_centroid[1] - cut_xywh[1] - int(mask_resize.shape[0] / 2) + half_padding
            bubble_start_w = bub_centroid[0] - cut_xywh[0] - int(mask_resize.shape[1] / 2) + half_padding

            # 원본 이미지에서 배경이 될 부분을 뽑습니다.
            roi = cut_bg[bubble_start_h:bubble_start_h + mask_resize.shape[0],
                  bubble_start_w:bubble_start_w + mask_resize.shape[1]]

            # 마스크 이용해서 오려내기
            masked_fg = cv2.bitwise_and(bub_fg_resize, bub_fg_resize, mask=mask_resize)
            masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # 이미지 합성
            added = masked_fg + masked_bg
            cut_bg_copy = cut_bg.copy()
            cut_bg_copy[bubble_start_h:bubble_start_h + mask_resize.shape[0],
            bubble_start_w:bubble_start_w + mask_resize.shape[1]] = added

            bubble_scope_cuts.append(cut_bg_copy)

            self.page_count += 1

        return bubble_scope_cuts


# 폴리곤으로 컷별 분리
def plus_cut(img, polygons):
    cuts = []
    #opencv 비트 연산으로 이미지 합성
    for idx, polygon in enumerate(polygons):
        croped = img.copy()
        mask = np.zeros(croped.shape[:2], np.uint8)
        bg = np.ones_like(croped, np.uint8) * 255
        for poly in polygons[:idx + 1]:
            cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        cv2.bitwise_not(bg, bg, mask=mask)
        cut = bg + dst
        cuts.append(cut)
    return cuts


# 컷 정렬
def sort_cut(padding_img_list, contours, read):
    height = padding_img_list[0].shape[0]
    width = padding_img_list[0].shape[1]
    n = 0
    sort_point = []
    # 좌측 최상단 좌표로 정렬
    for contour in contours:
        cx = contour.min(axis=0)[0][0] // int(width / 5)
        cy = contour.min(axis=0)[0][1] // int(height / 5)
        sort_point.append([cx, cy, n])
        n += 1
    # 만화책 좌상우하, 우상좌하 설정
    if read == 'L':
        sort_point.sort(key=lambda x: (x[1], x[0]))
    elif read == 'R':
        sort_point.sort(key=lambda x: (x[1], -x[0]))
    sort_point = np.array(sort_point)
    # 페이지에 말풍선이 하나도 없을 경우 예외사항 처리
    if not contours:
        return []
    else:
        index = sort_point[:, 2].tolist()
        sort_contours = [contours[i] for i in index]
        return sort_contours


# 학습 모델 적용
def cut_model(img_list, model_name):
    img_input = []
    # 모델 입력 전처리
    for img in img_list:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_res = cv2.resize(img_gray, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        img_input.append(img_gray_res / 255)

    img_input = np.asarray(img_input)
    img_input = img_input.reshape(img_input.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
    # 모델 적용
    model = load_model(model_name)
    predict = model.predict(img_input).reshape(len(img_input), IMAGE_SIZE, IMAGE_SIZE)
    # 출력 이미지 전처리
    labels = [np.around(label) * 255 for label in predict]
    labels = [cv2.resize(label, dsize=img_gray.shape[::-1], interpolation=cv2.INTER_AREA) for label in labels]
    return labels


# 페이지 별 폴리곤, 컷 제작
def make_polygons(padding_img_list, labels, read):
    cuts_list = []
    polygons_list = []
    for idx, label in enumerate(labels):
        label = np.asarray(label, dtype=np.uint8)
        contours, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 컷 정렬

        contours = sort_cut(padding_img_list, contours, read)
        polygons = [contour.reshape(contour.shape[0], 2) for contour in contours]
        cuts_list.append(plus_cut(padding_img_list[idx], polygons))
        polygons_list.append(polygons)
    return cuts_list, polygons_list


# 말풍선 컷 매칭
def bubble_cent(cuts_polygons, bubble_polygons):
    bubble_cents = []
    for idx, page in enumerate(cuts_polygons):
        bubble_page_cent = []
        for cut in page:
            bubble_cut_cent = []
            for i in bubble_polygons[idx]:

                M = cv2.moments(i)
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                # 말풍선 중심점을 이용 컷별 매칭
                if cv2.pointPolygonTest(cut, (cX, cY), True) >= 0:
                    bubble_cut_cent.append(i)
            bubble_page_cent.append(bubble_cut_cent)
        bubble_cents.append(bubble_page_cent)
    return bubble_cents


# 말풍선 확대 효과
def bubble_effect(cuts_list, bubbles_list, bubble_cents):
    # 말풍선 크기 조절 범위
    scope_list = np.linspace(1, 1.6, 12)
    final_list = []

    # 이미지 말풍선 비트 연산으로 합성
    for i, row_img in enumerate(cuts_list):
        page_img_list = []

        # 페이지에 말풍선이 하나도 없을 경우 예외사항 처리
        if not bubble_cents[i][0]:
            no_bubble = []
            for img in row_img:
                no_bubble.append([img])
            final_list.append(no_bubble)
        else:
            for j, row_cut_img in enumerate(row_img):
                bubble_img = bubbles_list[i][-1]
                polygons = bubble_cents[i][j]
                for polygon in polygons:
                    bubble_img_list = []
                    for scope in scope_list:
                        # 말풍선 영역 자르기
                        x, y, w, h = cv2.boundingRect(polygon)
                        roi_x = int(x + ((1 - scope) * w) * 0.5)
                        roi_y = int(y + ((1 - scope) * h) * 0.5)
                        roi = row_cut_img[roi_y: roi_y + int(scope * h), roi_x:roi_x + int(scope * w)].copy()
                        bubble_roi = bubble_img[y:y + h, x:x + w].copy()
                        res_bubble_roi = cv2.resize(bubble_roi, (roi.shape[1], roi.shape[0]))
                        pts = polygon - polygon.min(axis=0)

                        # 마스크 설정
                        mask = np.zeros(bubble_roi.shape[:2], np.uint8)
                        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

                        res_mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
                        res_mask_inv = cv2.bitwise_not(res_mask)

                        # 마스크로 자르기
                        cut_bubble = cv2.bitwise_and(res_bubble_roi, res_bubble_roi, mask=res_mask)
                        cut_bg = cv2.bitwise_and(roi, roi, mask=res_mask_inv)

                        # 이미지 합성
                        cut = cut_bg + cut_bubble
                        cut_img = row_cut_img.copy()

                        cut_img[roi_y: roi_y + int(scope * h), roi_x:roi_x + int(scope * w)] = cut
                        bubble_img_list.append(cut_img)
                    page_img_list.append(bubble_img_list)
            final_list.append(page_img_list)
    return final_list


def bubble_len(cuts_list, bubbles_list, bubble_cents):
    bub_len_list = []
    for i, row_img in enumerate(cuts_list):
        page_len = []
        if not bubble_cents[i][0]:
            no_bubble = []
            for img in row_img:
                no_bubble.append(0)
            bub_len_list.append(no_bubble)
        else:
            for j, row_cut_img in enumerate(row_img):
                img_shape = row_cut_img.shape[0]
                bubble_img = bubbles_list[i][-1]
                polygons = bubble_cents[i][j]
                if not polygons:
                    page_len.append(0)
                else:
                    for polygon in polygons:
                        # 말풍선 영역 자르기
                        x, y, w, h = cv2.boundingRect(polygon)
                        bubble_roi = bubble_img[y:y + h, x:x + w].copy()
                        pts = polygon - polygon.min(axis=0)
                        # 마스크 설정
                        mask = np.zeros(bubble_roi.shape[:2], np.uint8)
                        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
                        mask_inv = cv2.bitwise_not(mask)
                        # 마스크로 자르기
                        cut_bubble = cv2.bitwise_and(bubble_roi, bubble_roi, mask=mask)
                        gray_bubble = cv2.cvtColor(cut_bubble, cv2.COLOR_BGR2GRAY)
                        binarizer = preprocessing.Binarizer(threshold=150)
                        bub_binary = binarizer.transform(255 - (mask_inv + gray_bubble))
                        bub_len = np.sum(bub_binary) / img_shape
                        bub_len = np.round(bub_len, 2)
                        page_len.append(bub_len)
            bub_len_list.append(page_len)
    return bub_len_list

# 페이지 별 컷 최종 함수
def make_page_cut(img_list, read):
    cuts_list = []
    cut_labels = cut_model(img_list, 'model/best_gray_model_2.h5')
    bubble_labels = cut_model(img_list, 'model/bubble_gray_model.h5')

    # 이미지 패딩 적용
    hei = int(img_list[0].shape[0] * 0.15)
    wid = int(img_list[0].shape[1] * 0.15)

    # 이미지 데이터 white 패딩
    padding_cut_labels = [cv2.copyMakeBorder(img, hei, hei, wid, wid, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                          for img in cut_labels]
    padding_bubble_labels = [cv2.copyMakeBorder(img, hei, hei, wid, wid, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                             for img in bubble_labels]
    # 라벨 데이터 black 패딩
    padding_img_list = [cv2.copyMakeBorder(img, hei, hei, wid, wid, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                        for img in img_list]

    # 순서대로 함수 적용
    cuts_list, cuts_polygons = make_polygons(padding_img_list, padding_cut_labels, read)
    bubbles_list, bubble_polygons = make_polygons(padding_img_list, padding_bubble_labels, read)
    bubble_cents = bubble_cent(cuts_polygons, bubble_polygons)
    final_list = bubble_effect(cuts_list, bubbles_list, bubble_cents)

    bub_len_list = bubble_len(cuts_list, bubbles_list, bubble_cents)
    return final_list, bub_len_list
# ================================================================================================================


# =================================================== 수정된 부분 ====================================================
# [이미지,글자수]의 리스트를 받아 영상으로 만들고 저장하는 함수
def view_seconds(image_list, t_c, ani_effect, tran_effect, text_bubble_len_list):
    # 영상이름 오늘 날자와 시간으로 지정
    nowdate = datetime.datetime.now()
    daytime = nowdate.strftime("%Y-%m-%d_%H%M%S")
    # 영상 저장 위치 설정
    video_name = 'ani/' + daytime + '.mp4'
    out_path = 'media/' + video_name
    # video codec 설정
    fourcc = cv2.VideoWriter_fourcc(*'AVC1')

    # 여기서부터 컷과 책 형태 분리 하기
    # toon 형식
    if t_c == 'T':
        wid, hei = 2300, 2300
        fps = 35.0
        video = cv2.VideoWriter(out_path, fourcc, fps, (wid, hei))
        back_image = np.zeros((hei, wid, 3), np.uint8)
        # 3중 리스트로 되어있음
        for idx, image in enumerate(image_list):
            for i, j in enumerate(image):
                bubble_len = text_bubble_len_list[idx][i]
                if bubble_len > 0.09:
                    fpp = 80
                elif bubble_len < 0.05:
                    fpp = 20
                else:
                    fpp = 50
                # 말풍선 효과 넣었을 때
                if ani_effect == 'B':
                    # 이미지 resize 하여서 2300, 2300 중간위치에 넣기
                    for k in j:
                        cols, rows, channel = k.shape
                        shape_list = [2300 / cols, 2300 / rows]
                        k = cv2.resize(k, (0, 0), fx=min(shape_list), fy=min(shape_list), interpolation=cv2.INTER_CUBIC)
                        cols, rows, channel = k.shape
                        space_width = int((wid - rows) / 2)
                        space_height = int((hei - cols) / 2)
                        back_image = np.zeros((hei, wid, 3), np.uint8)
                        back_image[space_height:space_height + cols, space_width:space_width + rows] = k
                        video.write(back_image)
                    # 중간 이미지 60개 추가--나중에 이미지 글자 픽셀값으로 변경 될 수있음
                    for _ in range(fpp):
                        video.write(back_image)
                    # 거꾸로 이미지 resize 하여서 2300, 2300 중간위치에 넣기
                    for l in j[::-1]:
                        cols, rows, channel = l.shape
                        shape_list = [2300 / cols, 2300 / rows]
                        l = cv2.resize(l, (0, 0), fx=min(shape_list), fy=min(shape_list), interpolation=cv2.INTER_CUBIC)
                        cols, rows, channel = l.shape
                        space_width = int((wid - rows) / 2)
                        space_height = int((hei - cols) / 2)
                        back_image = np.zeros((hei, wid, 3), np.uint8)
                        back_image[space_height:space_height + cols, space_width:space_width + rows] = l
                        video.write(back_image)
                    last_frame = back_image
                # 말풍선 효과 안넣었을 때
                else:
                    # 이미지 resize 하여서 2300, 2300 중간위치에 넣기
                    image_hei, image_wid = j[0].shape[:2]
                    shape_list = [2300 / image_hei, 2300 / image_wid]
                    img_result = cv2.resize(j[0], (0, 0), fx=min(shape_list), fy=min(shape_list),
                                            interpolation=cv2.INTER_CUBIC)
                    back_image = np.zeros((hei, wid, 3), np.uint8)
                    cols, rows, channel = img_result.shape
                    space_width = int((wid - rows) / 2)
                    space_height = int((hei - cols) / 2)
                    # 말풍선 효과 길이만큼 보여주기- 나중에 이미지 글자 픽셀값으로 변경 될 수있음
                    each_image_duration = (len(j) * 2) + fpp
                    for k in range(each_image_duration):
                        back_image[space_height:space_height + cols, space_width:space_width + rows] = img_result
                        video.write(back_image)
                    last_frame = back_image

                # 여기서부터는 영상 전환 효과
                # image_list가 마지막이 아니고 image가 마지막일경우
                if (i + 1 == len(image)) and (idx + 1 < len(image_list)):
                    imag = image_list[idx + 1][0][0]
                # image가 마지막이 아니고 같은 컷이 아닐경우
                elif (i + 1 < len(image)) and (j[0].shape[:2] != image[i + 1][0].shape[:2]):
                    imag = image[i + 1][0]
                # 그외의 경우 효과넣지 않기
                else:
                    continue

                # 효과로 보여줄 다음 이미지 resize
                cols, rows, channel = imag.shape
                shape_list = [2300 / cols, 2300 / rows]
                imag = cv2.resize(imag, (0, 0), fx=min(shape_list), fy=min(shape_list), interpolation=cv2.INTER_CUBIC)
                image_hei, image_wid = imag.shape[:2]
                back_image = np.zeros((hei, wid, 3), np.uint8)
                space_width = int((wid - image_wid) / 2)
                space_height = int((hei - image_hei) / 2)
                back_image[space_height:space_height + image_hei, space_width:space_width + image_wid] = imag

                for p in range(1, int(fps + 1)):
                    frame = np.zeros((wid, hei, 3), dtype=np.uint8)
                    # 왼쪽으로...
                    if tran_effect == 'Lt':
                        dx = int((wid / fps) * p)
                        frame[:, 0:wid - dx, :] = last_frame[:, dx:wid, :]
                        frame[:, wid - dx:wid, :] = back_image[:, 0:dx, :]

                    # 오른쪽으로...
                    elif tran_effect == 'Rt':
                        dx = int((wid / fps) * p)
                        frame[:, 0:dx, :] = back_image[:, wid - dx:wid, :]
                        frame[:, dx:wid, :] = last_frame[:, 0:wid - dx, :]

                    # 위로...
                    elif tran_effect == 'U':
                        dx = int((hei / fps) * p)
                        frame[0:hei - dx, :, :] = last_frame[dx:hei, :, :]
                        frame[hei - dx:hei, :, :] = back_image[0:dx, :, :]

                    # 디졸브 효과
                    elif tran_effect == 'D':
                        alpha = p / fps
                        frame = cv2.addWeighted(last_frame, 1 - alpha, back_image, alpha, 0)

                    video.write(frame)
        # 객체를 반드시 종료시켜주어야 한다
        video.release()
    # comic 만화책 형식일때
    else:
        # 가장 첫번째 이미지 사이즈 받아서 사용
        hei, wid, channel = image_list[0][0][0].shape
        fps = 35.0
        video = cv2.VideoWriter(out_path, fourcc, fps, (wid, hei))
        # 맨 처음 하얀 화면 만들어서 비디오 기록
        frame = np.ones((hei, wid, 3), dtype=np.uint8) * 255
        last_frame = frame
        video.write(frame)
        # 3중 리스트로 되어있음
        for idx, image in enumerate(image_list):
            for i, j in enumerate(image):
                bubble_len = text_bubble_len_list[idx][i]
                if bubble_len > 1.5:
                    fpp = 80
                elif bubble_len < 0.5:
                    fpp = 20
                else:
                    fpp = 50
                # 컷 등장 디졸브 효과
                for p in range(1, int(fps + 1)):
                    alpha = p / fps

                    frame = cv2.addWeighted(frame, 1 - alpha, j[0], alpha, 0)
                    video.write(frame)
                # 말풍선 효과 넣었을 때
                if ani_effect == 'B':
                    for u, k in enumerate(j):

                        video.write(k)
                        imglast = k
                    for _ in range(fpp):
                        video.write(imglast)
                    for l in j[::-1]:
                        video.write(l)
                        la_img = l
                    last_frame = la_img
                # 말풍선 효과 안넣었을 때
                else:
                    each_image_duration = (len(j) * 2) + fpp
                    for k in range(each_image_duration):
                        video.write(j[0])
                    last_frame = j[0]

            # 여기서부터는 영상 전환 효과
            # 마지막 이미지에는 효과 넣지 않기
            if idx + 1 >= len(image_list):
                continue

            for p in range(1, int(fps + 1)):
                frame = np.ones((hei, wid, 3), dtype=np.uint8) * 255
                # 왼쪽으로...
                if tran_effect == 'Lt':
                    dx = int((wid / fps) * p)

                    frame[:, 0:wid - dx, :] = last_frame[:, dx:, :]

                # 오른쪽으로...
                elif tran_effect == 'Rt':
                    dx = int((wid / fps) * p)
                    frame[:, dx:, :] = last_frame[:, 0:wid - dx, :]

                # 위로...
                elif tran_effect == 'U':
                    dx = int((hei / fps) * p)
                    frame[0:hei - dx, :, :] = last_frame[dx:, :, :]

                # 디졸브 효과
                elif tran_effect == 'D':
                    alpha = p / fps
                    frame = cv2.addWeighted(last_frame, 1 - alpha, frame, alpha, 0)

                video.write(frame)
        # 객체를 반드시 종료시켜주어야 한다
        video.release()
        # ==========================================================================================================

    # 영상 저장 위치 반환
    return video_name
