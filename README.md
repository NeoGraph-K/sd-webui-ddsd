# sd-webui-ddsd
자동으로 동작하는 후보정 작업 확장.

## What is
### Upscale
이미지를 특정 크기로 잘라내어 타일별 업스케일을 하는 도구. 업스케일시 VRAM을 적게 소모.
#### Upscale How to use
1. 크기를 키울때 사용할 upscaler 모델 선택
2. 크기를 키울 배수 선택
3. 가로, 세로를 내가 단일로 생성할 수 있는 이미지의 최대 크기로 선택(이미지 생성 속도를 최대한 빠르게 하기 위하여)
    1. 가로 또는 세로중 한개를 0으로 세팅시 업스케일만 동작(세부 구조를 디테일하게하는 인페인팅이 동작하지 않음)
4. before running 체크
    1. 체크시 업스케일을 먼저 돌려서 인페인팅의 퀄리티 상승. 단, 인페인팅시 더 많은 VRAM 요구
5. 생성!
### Detect Detailer
특정 키워드로 이미지를 탐색 후 인페인팅하는 도구.
#### Detect Detailer How to use
0. 인페인팅의 범위 제한(I2I 전용)
    1. Inner 옵션은 I2I의 인페인팅에서 칠한 범위 내부만 이미지를 탐색
    2. Outer 옵션은 I2I의 인페인팅에서 칠한 범위 외부만 이미지를 탐색
1. 탐색 키워드 작성
    1. 탐색할 키워드를 작성(face, person 등등)
        1. 탐색할 키워드는 문장형도 가능(happy face, running dog)
        2. 탐색할 키워드를 .으로 분할 가능(face. arm, face. chest)
    2. 탐색할 키워드에 사용 가능한 추가 옵션 존재
        1. &lt;area:type&gt;을 이용하여 특정 범위 탐색 가능
            1. 범위 종류는 left, right, top, bottom, all이 존재
        2. &lt;file:filename&gt;을 이용하여 특정 파일 탐색 가능
            1. 특정 파일의 위치는 models/ddsdmask
        3. &lt;model:type&gt;을 이용하여 특정 모델 탐색 가능
            1. type은 face_media_full, face_media_short와 파일명이 존재
            2. 파일은 models/yolo에 위치
        4. &lt;type1:type2:dilation:confidence&gt; 같이 type1과 type2외에 dilation과 confidence도 추가 입력 가능
            1. confidence는 model 타입에서만 사용되는 값
    3. 탐색한 범위를 AND, OR, XOR, NAND, NOR 등의 게이트 옵션으로 연산 가능
        1. face OR (body NAND outfit) -> 괄호안의 body NAND outfit을 먼저 한 후에 face와 OR 연산을 동작
        2. 괄호는 최대한 적게 이용. 많이 이용시 많은 VRAM 소모.
        3. 동작은 왼쪽에서 오른쪽으로 순차적 동작.
    4. 탐색할 키워드에 옵션으로 여러가지 옵션 조절 가능
        1. face:0:0.4:4 OR outfit:2:0.5:8
        2. 순서대로 탐색할 프롬프트, SAM 탐색 레벨(0-2), 민감도(0-1), 팽창값(0-512)을 가짐
        3. 값을 생략하면 초기값으로 세팅
2. 긍정 프롬프트 입력
    1. 인페인팅시 동작시킬 긍정 프롬프트 입력
3. 부정 프롬프트 입력
    1. 인페인팅시 동작시킬 부정 프롬프트 입력
4. Denoising, CFG, Steps, Clip skip, Ckpt, Vae 수정
    1. 인페인팅시 동작에 영향을 주는 옵션
5. Split Mask 옵션 체크
    1. 체크시 마스크가 떨어져 있는것이 존재한다면 따로 인페인팅.
        1. 따로 인페인팅시 퀄리티 상승. 하지만 더 많은 인페인팅을 요구하여 생성속도 하락.
6. Remove Area 옵션 체크
    1. Split Mask 옵션이 Enable 되어야만 동작
    2. 분할 인페인팅시 일정 크기 이하의 면적은 인페인팅에서 제외
6. 생성!
### Postprocessing
최종적으로 생성된 이미지에 가하는 후보정
#### Postprocessing How to use
1. 가하고자 하는 후보정을 선택
2. 생성!
### Watermark
이미지 생성 최종본에 자신의 증명을 기입하는 기능
#### Watermark How to use
1. 기입할 증명의 종류 선택(글자, 이미지)
2. 선택한 종류를 입력
3. 선택한 종류의 크기와 위치를 지정
4. Padding으로 해당 위치에서 얼만큼 떨어져 있을지 설정
5. Alpha로 얼만큼 투명할지 결정
6. 생성!

### Video
<iframe width="1237" height="696" src="https://www.youtube.com/embed/9wfZyJhPPho" title="Stable Diffusion - DDSD 확장 기능  (No - Talking)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Installation
1. 다운로드 [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)와 [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)
    1. 자신이 가진 WebUI와 동일한 버전의 `CUDA`와 `cuDNN`버전으로 설치
        1. 이것은 다운로드를 편하게 하기위한 구글링크. [CUDA 117](https://drive.google.com/file/d/1HRTOLTB44-pRcrwIw9lQak2OC2ohNle3/view?usp=share_link)와 [cuDNN](https://drive.google.com/file/d/1QcgaxUra0WnCWrCLjsWp_QKw1PKcvqpj/view?usp=share_link)
    2. `CUDA` 설치 후 해당 폴더에 `cuDNN` 덮어쓰기
    3. 일정 버전은 Easy Install을 지원. `CUDA`와 `cuDNN` 불필요.
        1. 지원버전 (torch == 1.13.1+cu117, torch==2.0.0+cu117 , torch==2.0.0+cu118)
2. 확장탭에서 설치 `https://github.com/NeoGraph-K/sd-webui-ddsd` 또는 다운로드 후 `extension/` 에 풀어넣기
3. WebUI를 완전히 재시작

## Credits

dustysys/[ddetailer](https://github.com/dustysys/ddetailer)

AUTOMATIC1111/[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

facebookresearch/[Segment Anything](https://github.com/facebookresearch/segment-anything)

IDEA-Research/[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

IDEA-Research/[Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)

continue-revolution/[sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything)

Bing-su/[adetailer](https://github.com/Bing-su/adetailer)
