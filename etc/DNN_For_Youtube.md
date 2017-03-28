### 원문 https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=4&cad=rja&uact=8&ved=0ahUKEwj23NGD0fjSAhVjOpoKHeB6A3oQFgg6MAM&url=https%3A%2F%2Fresearch.google.com%2Fpubs%2Farchive%2F45530.pdf&usg=AFQjCNH3ZCLR2fpu57Cf7004q3YRrJ0N7A&sig2=FfI2X-UmkNox5Y7G3YpE5A
### 참고 http://keunwoochoi.blogspot.kr/2016/09/deep-neural-networks-for-youtube.html

### 번역 & 요약본입니다
***



# Deep Neural Networks for YouTube Recommendations
## 2016.09.15

## 1. 서문
유튜브 추천시스템 구축은 다음 세가지 관점에서 매우 쉽지 않음

* Scale  
기존에 많은 추천 알고리즘은 유튜브 사이즈(방대한 영상 수 등)에 맞지 않음  
고도의 분산 학습 알고리즘이 필요  

* Freshness  
실시간으로 새로운 비디오가 많이 등록됨  
새로운 컨텐츠와 기존 컨텐츠의 추천 발란스가 중요함  

* Noise  
Noise 요소 : 데이터의 희소성(sparsity), 관찰 불가능한 외부요인, 정답 없음(ground truth), 암시적인 피드백(implicit feedback)

구글과 마찬가지로 Youtube 도 deep-learning 을 learning 문제의 일반적인 방법론으로 전환하고 있다.
방대하게 많은 matrix factorization 기법에 비해, deep-learning 추천 연구는 상대적으로 적다.

## 2. 시스템 개요
개략적인 시스템 구조는 아래 그림과 같다.
두개의 핵심 네트워크는 다음과 같다

1. 후보 작품 선정 (cadidate generation)  
사용자의 유튜브 이력을 입력으로 추천할 후보비디오를 몇 백개로 추려냄  
CF(협력 필터링, collaborative filtering)을 사용하여 개인화 함  
사용자간의 유사도는 굵직한 피처(coarse feature)로 나타난다 (ex 시청한 비디오 ID, 검색 쿼리 분석, 인구통계 등)

2. 순위 매김 (ranking)  
위의 후보 중, best 추천 작품 몇개만 추려야 함   
이 네트워크에선 비디오에 점수를 부여해서 사용자에게 몇개(dozens)의 작품만 보여줌  
이 점수는 user-video 간의 다양한 요소로 부여함  

이 시스템은 다른 요소도 적용할 수 있게 설계되었다. (other candidate sources)  
ex) 기존 Youtube 시스템의 추천

성능을 평가하기 위해 A/B 테스트를 진행 하였으며, 클릭율, 시청시간, 그외 다양한 항목을 측정할 수 있었다.  
이 live A/B 테스트는 중요하다. 왜냐하면, A/B테스트는 오프라인 테스트와 연관관계가 항상 일치하지는 않기 때문이다.

## 3. 후보 작품 선정
기존에는 ranking loss를 이용한 matrix factorisation 을 사용했었음  
연구 초기에는 이 MF기법을 모방하여 사용자의 과거 시청이력만을 사용했었다.  
이 경우 비선형 일반화가 되는듯 보였다.

### 3.1 추천 분류 (Recommendation as Classification)
추천은 어마하게 많은 분류(multiclass) 문제라고 여겼다.  
요소  :
* 수많은 비디오 중, 한 사용자에 대한 비디오 재생시간과 context(흐름)
* 적극적 행위(좋아요,싫어요, 서베이 등)는 제외하고 묵시적인 피드백(implicit feedback)(예 재생완료 작품)만 고려함
    * 적극적 행위 배제 요인 : 전체 사용자 대비 적극적 행위(explicit feedback)은 매우 희소함

### 3.2 아키텍처
전체 아키텍쳐는 다음과 같다.

입력값 
* 사용자의 비디오 시청내역과, 검색 키워드 토큰, 나이, 성별, 지리등