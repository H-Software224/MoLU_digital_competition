# MoLU_digital_competition

## 프로젝트 설명
SW중심대학 디지털 경진대회_SW와 생성AI의 만남 : AI부문
+ 이 AI 경진대회에서는 5초 분량의 오디오 샘플에서 진짜 사람 목소리와 AI가 생성한 가짜 목소리를 정확하게 구분할 수 있는 모델을 개발하는 것이 목표입니다.
+ 이 작업은 보안, 사기 감지 및 오디오 처리 기술 향상 등 다양한 분야에서 매우 중요합니다.
+ 제공되는 베이스라인 코드에서는 음성 파일에서 MFCC(Mel Frequency Cepstral Coefficients)로 특징을 추출하여 MLP(다층 퍼셉트론) 모델을 학습시키고 추론하는 과정을 포함하고 있으며, MFCC가 아닌 다른 방법론 역시 충분히 적용해볼 수 있습니다.

## 프로젝스 과정 설명
1. 모듈을 import 합니다.
librosa<br>

sklearn<br>
pandas<br>
torch<br>
3. Config을 통해서 parameter, hyperparamter를 적용합니다.
4. data processing으로 mfcc 기법을 이용합니다.
5. model를 mlp로 정하여서 모델링합니다.
6. 이 model를 학습하며, 결과를 도출합니다.
