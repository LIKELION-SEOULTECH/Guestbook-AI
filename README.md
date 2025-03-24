# Guestbook-AI

# 해당 프로젝트에서 사용한 감정 모델 (kobert_model.onnx)


|count|Emotion|
|----|-----|
|fear|1386|
|surprise|1755|
|angry|3263|
|sadness|4548|
|neutral|3253|
|happiness|4548|
|disgust|2321|

- 데이터 예시

||Sentence	|Emotion|
|--|---|--|
|0|	헐! 나 이벤트에 당첨 됐어.	|happiness|
|1|	이번 달에 또 급여가 깎였어! 물가는 오르는데 월급만 자꾸 깎이니까 너무 화가 나....	|happiness|
|2|	내가 좋아하는 인플루언서가 이벤트를 하더라고. 그래서 그냥 신청 한번 해봤지. |happiness|
|3|	한 명 뽑는 거였는데, 그게 바로 내가 된 거야.	|happiness|
|4|	에피타이저 정말 좋아해. 그 것도 괜찮은 생각인 것 같애.	|neutral|
|4|	나 요즘 너무 우울해 죽겠어.	|sadness|


* 데이터 출처 : https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=263


-----


## 다른 모델 
### 1. emotion_model.onnx
- 데이터 분포
  
|count|Emotion|
|----|-----|
|불안|10433|
|분노|10417|
|상처|10150|
|슬픔|10128|
|당황|9804|
|기쁨|7339|

- 데이터 예시
  
||Sentence	|Emotion|
|--|---|--|
|0|	일은 왜 해도 해도 끝이 없을까? 화가 난다.그냥 내가 해결하는 게 나아. 남들한테...	|분노|
|1|	이번 달에 또 급여가 깎였어! 물가는 오르는데 월급만 자꾸 깎이니까 너무 화가 나....	|분노|
|2|	회사에 신입이 들어왔는데 말투가 거슬려. 그런 애를 매일 봐야 한다고 생각하니까 스... |분노|
|3|	직장에서 막내라는 이유로 나에게만 온갖 심부름을 시켜. 일도 많은 데 정말 분하고 ...	|분노|
|4|	얼마 전 입사한 신입사원이 나를 무시하는 것 같아서 너무 화가 나.상사인 나에게 먼...	|분노|

* 데이터 출처 : https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86

## 다른 모델 
### 2. emotion_model_include_neutral.onnx
이 모델은 데이터 수가 많지 않고 말투가 커뮤니티의 성격이 짙어 일반적인 구어체에 제대로 된 성능을 보이지 못한다.
- 데이터 분포
  
|count|Emotion|
|---|----|
|행복	|6037|
|놀람	|5898|
|분노	|5665|
|공포	|5468|
|혐오	|5429|
|슬픔	|5267|
|중립	|4830|

- 데이터 예시

||Sentence|	Emotion|
|--|---|--|
|0|	언니 동생으로 부르는게 맞는 일인가요..??	|공포|
|1|	그냥 내 느낌일뿐겠지?	|공포|
|2|	아직너무초기라서 그런거죠?	|공포|
|3|	유치원버스 사고 낫다던데	|공포|
|4|	근데 원래이런거맞나요	|공포|

- 데이터 출처 : https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=270
