#  BLIP 기반 실시간 영상 캡셔닝 시스템

## 프로젝트 소개
이 프로젝트는 Salesforce의 BLIP(이미지 캡셔닝) 딥러닝 모델을 활용하여,
실시간 웹캠 스트림 또는 비디오 파일에서 프레임별로 자동으로 이미지를 캡션하고,
이를 한글로 번역하여 영상 위에 오버레이하는 Python 기반 영상 분석 도구입니다.

## 주요 기능
- 실시간 웹캠 기반 캡셔닝 (+배경 마스킹)
- 비디오 파일 분석 및 캡션 오버레이
- 영문 캡션 → 한글 번역 통합

## 설치 방법
1. Python 3.8 이상 설치
2. 필수 패키지 설치

   ```
   pip install -r requirements.txt
   ```

## 파일 설명
`BLIP_VIDEO`: 비디오 파일 기반  캡셔닝

`BLIP_CAM`: 캠 기반 실시간 캡셔닝(+배경 마스킹)

## 결과 예시
#### BLIP_VIDEO: 비디오 파일 기반 캡셔닝
<img src="https://github.com/user-attachments/assets/9d47accb-def6-4996-b3a6-c4e2d7f987c5" width="1000">

#### BLIP_CAM: 실시간 웹캠 캡셔닝 (+배경 마스킹)
<img src="https://github.com/user-attachments/assets/8798a070-a60c-415c-8a65-64822639f1ab" width="1000">
