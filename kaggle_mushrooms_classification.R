# 1. 분석 설명 : 기계학습 - 지도학습 - 분류분석(classification) ---------
# Data : Mushrooms.csv
# 사용한 머신러닝 기법 : LASSO, RandomForest, ElasticNet, Boost
# 가능하다면 앙상블이나 메타분석까지 실행해보자 


## 이 프로젝트 하면서 내가 배운 점 / 궁금했던 점 -----------
# 1) 모든 변수가 levels가 있는 factor 변수들인데도 classification 이 돌아갈까?
# 2) 군집분석이랑 분류분석의 차이는?
# (지도학습, 강화학습 이런 카테고리의 차이 말고 원리의 차이와 결론을 어떻게 사용하는 것인지)
# 3)


# 2. Algorithm --------------------------------------------------------------------


# GroundWork ----------------------------------------------------------------------
rm(list=ls())

getwd()
setwd("C:/Users/LG/Documents/Study-R/Data")
getwd()

# 패키지 장착 
library(dplyr)
library(descr)
library(DT)
library(ggplot2)
library(ISLR)
library(MASS)
library(glmnet)
library(randomForest)
library(gbm)
library(rpart)
library(boot)

# Load Data Set ---------------
mushrooms <- read.csv("mushrooms.csv",
                      header = T)

# 데이터 확인 Checking Data ---------------
str(mushrooms)
# 데이터 속성 data frame / 관측치 8124 / 변수 23개 임을 확인
# 변수 타입이 모두 factor 에 levels 적용되어 있음을 확인 

summary(mushrooms$veil.type)
# veil.type 변수는 모두 1 level 이라 무의미한 변수이므로 제거 함.
mushrooms <- mushrooms[, -17]

head(mushrooms)
DT::datatable(mushrooms)

summary(mushrooms)

levels(mushrooms$class)

mushrooms$class <- factor(mushrooms$class,
                          levels = c("p", "e"))

levels(mushrooms$class)

### 고민해볼 포인트 `stalk-root`(줄기 뿌리) 변수에서 `?`(missing, 누락) 값을 결측치로 제거 할 것인지 말 것인지  --------------

#### 문제의 복잡도 구하기
# 단 mushrooms 데이터에는 수치형 설명변수(입력 변수)가 존재하지 않아서 실행되지 않는다.
# x <- model.matrix( ~ . - class, mushrooms)
# dim(x)

# EDA Visualization ------------------
# 죄다 factor 인데 어떤 시각화하지....? 
# GGally::ggpairs()
# 데이터가 질적 자료면 막대그래프,
# 데이터가 양적 자료이면 히스토그램, 상자그림, 산점도 등을 한꺼번에 그려주는 함수이다.
install.packages("GGally")
library(GGally)

GGally::ggpairs(mushrooms)

# 지금 분석중인 mushrooms 데이터는 모든 변수가 질적 자료라 막대 그래프만 출력됨.
# 그럼 시각화는 막대그래프만 제한될까...? 의미있는 시각화에는 뭐가 있을까?

# 일단 제일 levels 가 다양한 gill.color / cap.color 변수만 골라서 막대그래프로 나타내보자

mushrooms %>%
  ggplot(aes(gill.color)) +
  geom_bar()

mushrooms %>%
  ggplot(aes(cap.color)) +
  geom_bar()

# 따라하며 배우는 데이터 과학 - 데이터 종류에 따른 분석 기법 파트 참조 
# 일단 class 변수의 막대 그래프 시각화 - class 변수에 식용과 독성 버섯의 분포 확인
mushrooms %>%
  ggplot(aes(class)) +
  geom_bar()

mushrooms %>%
  group_by(class) %>%
  ggplot(aes(cap.color, fill = class)) +
  geom_bar(position = "dodge")

# 빈도수와 비율 
CrossTable(mushrooms$class)


# [출처] 데이터 과학 입문자를 위한 R, 재리드 랜더 지음, 고석범 옮김, 에이콘, p332~333
# 출처2 이부일 박사님 페북ㅋㅋㅋㅋ

# 모자이크 플롯(Mosaicplot)
# 질적 자료(factor) 다변량 데이터에를 표현하는데 적합한 그래프이다.
mosaicplot( ~ gill.color + class,
            data = mushrooms,
            color=T,
            cex=1.2)

mosaicplot( ~ cap.color + class,
            data = mushrooms,
            color=T,
            cex=1.2)

# Classification Start  ----------------
#### 모델 순서 : 
# 1. 로지스틱 회귀 모형(Logistic Regression Model)
# 2. Random Forest
# 3. gbm
# 4. LASSO 

# 데이터셋 나누기 --------------------------------
#### train : validation : test == 60 : 20 : 20 으로 나눔   

set.seed(0124)                # seed 는 보통 연구한 년월을 쓰는 등 여러 방법이 있음.

n <- nrow(mushrooms)
idx <- 1:n                    # 총 관측치 개수 인덱싱

training.idx <- sample(idx, n * .60)   # Random 하게 전체 데이터에서 60% 샘플링
idx <- setdiff(idx, training.idx)    
# 전체 idx에서 training_idx 제외한 나머지 idx를 다시 idx 변수에 저장

validation.idx <- sample(idx, n * .20)
test.idx <- setdiff(idx, validation.idx)

# 샘플링 된 데이터 갯수들 확인 
length(training.idx)
length(validation.idx)
length(test.idx)

# 순서대로 훈련, 검증, 테스트 데이터 
training <- mushrooms[training.idx, ]
validation <- mushrooms[validation.idx, ]
test <- mushrooms[test.idx, ]

#   Decision Tree ------------------------

cvr_tr <- rpart(class ~ . , data = training)
cvr_tr

summary(cvr_tr)

opar <- par(mfrow = c(1,1),
            xpd = NA)

plot(cvr_tr)
text(cvr_tr, use.n=TRUE)
par(opar)

# 1. 로지스틱 회귀  ------------------------
# 반응변수(class)의 범주가 2개(levels == 2) 이므로 이진형(binary) 이다.
# 따라서 후에 family = "binomial" 로 설정해줘야 한다.

mushrooms_glm <- glm(class,
                     data = training,
                     family = binomial)

# 변수가 모두 factor(질적 자료) 라서 안 돌아가는 건가 싶다 -----------







## 2. RandomForest  ----------------
set.seed(0124)

mushrooms_rf <- randomForest(class ~ . , training)
mushrooms_rf

importance(mushrooms_rf)
varImpPlot(mushrooms_rf)
# odor 과 spore.print.color 변수가 특히 유의함을 알 수 있다. 

predict(mushrooms_rf,
        newdata = validation)

predict(mushrooms_rf,
        newdata = validation, type = "prob")

## 3. gbm (Generalized Boosted Regression Models)------------------
set.seed(0124)

mushrooms_gbm <- training %>%
  mutate(class = ifelse(class == "e", 1, 0))

mushrooms_gbm$class

mushrooms_gbm <- gbm(class ~ . ,
                     data = mushrooms_gbm,
                     distribution = "bernoulli",
                     n.trees = 70000, cv.folds = 5, verbose = T)

(best_iter <- gbm.perf(mushrooms_gbm, method="cv"))

## 3. LASSO -----------------
xx <- model.matrix(class ~ . , mushrooms)
x <- xx[training.idx,]
y <- ifelse(training$class == "e", 1, 0)
dim(x)   
# 가변수(더미변수) 포함한 훈련 데이터의 갯수 / 4874개 관측치, 변수 96개 

mushrooms_glmnet_fit <- glmnet(x,y) # default LASSO, alpha=1
plot(mushrooms_glmnet_fit)          # 하단숫자: L1norm값, 상단 숫자:0이 아닌 모수의 갯수
# 19000개 관측치, 변수 101개 중에서 선택된 변수들이 얼마나 되는지 시각적으로 확인 

mushrooms_glmnet_fit                # % Dev: 현재 모형으로 설명되는 변이의 부분을 의미함

coef(mushrooms_glmnet_fit,
     s = 0.1737)
# eta = 0.2244 + 0.0338943*marital-status`Married-civ-spouse

mushrooms_cvfit <- cv.glmnet(x, y, family = "binomial")
plot(mushrooms_cvfit)

log(mushrooms_cvfit$lambda.min) 
# 최적의 예측력 : 예측력이 가장 좋을 때는 lamda가 가장 작을때임

log(mushrooms_cvfit$lambda.1se) 
# 해석 가능한 모형 

# lambda.1se 일 때 각 변수들의 계수들 출력 
coef(mushrooms_cvfit,
     s = mushrooms_cvfit$lambda.1se)

# lambda.min 일 때 각 변수들의 계수들 출력 
coef(mushrooms_cvfit,
     s = mushrooms_cvfit$lambda.min)


# 데이터의 모든 변수들 중에 영향력이 있다고 판별된 변수들의 갯수 출력 
length(which(coef(mushrooms_cvfit,
                  s="lambda.min")!=0))

length(which(coef(mushrooms_cvfit,
                  s="lambda.1se")!=0))


# Last Visualization ------------------

# ROC Curve, AUC Value
# TP FP TF FF 테이블도 출력 
# 적당한 시행횟수 출력해보기
# 라쏘의 경우 우선순위 높은 변수들 출력해보기 
# 랜덤 포레스트의 경우 연관성 정도는 모르지만 영향력이 강한 변수는 알 수 있음. 이것도 출력해보자 


# 결론 : 최종 선택 모델 ----------------
# 예측력과 설명력 둘 다 비교하기 


# 마크다운 작성하기 -----------









