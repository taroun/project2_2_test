from django import forms
from django.forms import ModelForm, HiddenInput

from animateapp.models import Animate


class AnimateForm(ModelForm):
    class Meta:
        model = Animate
        fields = ['image', 'ani', 'left_right', 'toon_comic', 'ani_effect', 'transition_effect']
        widgets = {
            # 생성시 안보이게 처리
            'ani': HiddenInput(),
            'image': forms.ClearableFileInput(attrs={'multiple': True}),
            # 'left_right': forms.Select(attrs={'id': 'flex_line0'}),
            # 'toon_comic': forms.Select(attrs={'id': 'flex_line1'}),
            # 'ani_effect': forms.Select(attrs={'id': 'flex_line2'}),
            # 'transition_effect': forms.Select(attrs={'id': 'flex_line3'}),
        }
        labels = {
            #생성시 라벨 변경
            'left_right': '만화책 방향',
            'toon_comic': '보고싶은 형식',
            'ani_effect': '영상효과',
            'transition_effect': '영상 전환 효과',
        }

