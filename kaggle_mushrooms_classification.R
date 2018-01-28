# 1. �м� ���� : ����н� - �����н� - �з��м�(classification) ---------
# Data : Mushrooms.csv
# ����� �ӽŷ��� ��� : LASSO, RandomForest, ElasticNet, Boost
# �����ϴٸ� �ӻ���̳� ��Ÿ�м����� �����غ��� 


## �� ������Ʈ �ϸ鼭 ���� ��� �� / �ñ��ߴ� �� -----------
# 1) ��� ������ levels�� �ִ� factor �������ε��� classification �� ���ư���?
# 2) �����м��̶� �з��м��� ���̴�?
# (�����н�, ��ȭ�н� �̷� ī�װ����� ���� ���� ������ ���̿� ����� ��� ����ϴ� ������)
# 3)


# 2. Algorithm --------------------------------------------------------------------


# GroundWork ----------------------------------------------------------------------
rm(list=ls())

getwd()
setwd("C:/Users/LG/Documents/Study-R/Data")
getwd()

# ��Ű�� ���� 
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

# ������ Ȯ�� Checking Data ---------------
str(mushrooms)
# ������ �Ӽ� data frame / ����ġ 8124 / ���� 23�� ���� Ȯ��
# ���� Ÿ���� ��� factor �� levels ����Ǿ� ������ Ȯ�� 

summary(mushrooms$veil.type)
# veil.type ������ ��� 1 level �̶� ���ǹ��� �����̹Ƿ� ���� ��.
mushrooms <- mushrooms[, -17]

head(mushrooms)
DT::datatable(mushrooms)

summary(mushrooms)

levels(mushrooms$class)

mushrooms$class <- factor(mushrooms$class,
                          levels = c("p", "e"))

levels(mushrooms$class)

### �����غ� ����Ʈ `stalk-root`(�ٱ� �Ѹ�) �������� `?`(missing, ����) ���� ����ġ�� ���� �� ������ �� ������  --------------

#### ������ ���⵵ ���ϱ�
# �� mushrooms �����Ϳ��� ��ġ�� ��������(�Է� ����)�� �������� �ʾƼ� ������� �ʴ´�.
# x <- model.matrix( ~ . - class, mushrooms)
# dim(x)

# EDA Visualization ------------------
# �˴� factor �ε� � �ð�ȭ����....? 
# GGally::ggpairs()
# �����Ͱ� ���� �ڷ�� ����׷���,
# �����Ͱ� ���� �ڷ��̸� ������׷�, ���ڱ׸�, ������ ���� �Ѳ����� �׷��ִ� �Լ��̴�.
install.packages("GGally")
library(GGally)

GGally::ggpairs(mushrooms)

# ���� �м����� mushrooms �����ʹ� ��� ������ ���� �ڷ�� ���� �׷����� ��µ�.
# �׷� �ð�ȭ�� ����׷����� ���ѵɱ�...? �ǹ��ִ� �ð�ȭ���� ���� ������?

# �ϴ� ���� levels �� �پ��� gill.color / cap.color ������ ��� ����׷����� ��Ÿ������

mushrooms %>%
  ggplot(aes(gill.color)) +
  geom_bar()

mushrooms %>%
  ggplot(aes(cap.color)) +
  geom_bar()

# �����ϸ� ���� ������ ���� - ������ ������ ���� �м� ��� ��Ʈ ���� 
# �ϴ� class ������ ���� �׷��� �ð�ȭ - class ������ �Ŀ�� ���� ������ ���� Ȯ��
mushrooms %>%
  ggplot(aes(class)) +
  geom_bar()

mushrooms %>%
  group_by(class) %>%
  ggplot(aes(cap.color, fill = class)) +
  geom_bar(position = "dodge")

# �󵵼��� ���� 
CrossTable(mushrooms$class)


# [��ó] ������ ���� �Թ��ڸ� ���� R, �縮�� ���� ����, ������ �ű�, ������, p332~333
# ��ó2 �̺��� �ڻ�� ��Ϥ�������

# ������ũ �÷�(Mosaicplot)
# ���� �ڷ�(factor) �ٺ��� �����Ϳ��� ǥ���ϴµ� ������ �׷����̴�.
mosaicplot( ~ gill.color + class,
            data = mushrooms,
            color=T,
            cex=1.2)

mosaicplot( ~ cap.color + class,
            data = mushrooms,
            color=T,
            cex=1.2)

# Classification Start  ----------------
#### �� ���� : 
# 1. ������ƽ ȸ�� ����(Logistic Regression Model)
# 2. Random Forest
# 3. gbm
# 4. LASSO 

# �����ͼ� ������ --------------------------------
#### train : validation : test == 60 : 20 : 20 ���� ����   

set.seed(0124)                # seed �� ���� ������ ����� ���� �� ���� ����� ����.

n <- nrow(mushrooms)
idx <- 1:n                    # �� ����ġ ���� �ε���

training.idx <- sample(idx, n * .60)   # Random �ϰ� ��ü �����Ϳ��� 60% ���ø�
idx <- setdiff(idx, training.idx)    
# ��ü idx���� training_idx ������ ������ idx�� �ٽ� idx ������ ����

validation.idx <- sample(idx, n * .20)
test.idx <- setdiff(idx, validation.idx)

# ���ø� �� ������ ������ Ȯ�� 
length(training.idx)
length(validation.idx)
length(test.idx)

# ������� �Ʒ�, ����, �׽�Ʈ ������ 
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

# 1. ������ƽ ȸ��  ------------------------
# ��������(class)�� ���ְ� 2��(levels == 2) �̹Ƿ� ������(binary) �̴�.
# ���� �Ŀ� family = "binomial" �� ��������� �Ѵ�.

mushrooms_glm <- glm(class,
                     data = training,
                     family = binomial)

# ������ ��� factor(���� �ڷ�) �� �� ���ư��� �ǰ� �ʹ� -----------







## 2. RandomForest  ----------------
set.seed(0124)

mushrooms_rf <- randomForest(class ~ . , training)
mushrooms_rf

importance(mushrooms_rf)
varImpPlot(mushrooms_rf)
# odor �� spore.print.color ������ Ư�� �������� �� �� �ִ�. 

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
# ������(���̺���) ������ �Ʒ� �������� ���� / 4874�� ����ġ, ���� 96�� 

mushrooms_glmnet_fit <- glmnet(x,y) # default LASSO, alpha=1
plot(mushrooms_glmnet_fit)          # �ϴܼ���: L1norm��, ��� ����:0�� �ƴ� ����� ����
# 19000�� ����ġ, ���� 101�� �߿��� ���õ� �������� �󸶳� �Ǵ��� �ð������� Ȯ�� 

mushrooms_glmnet_fit                # % Dev: ���� �������� �����Ǵ� ������ �κ��� �ǹ���

coef(mushrooms_glmnet_fit,
     s = 0.1737)
# eta = 0.2244 + 0.0338943*marital-status`Married-civ-spouse

mushrooms_cvfit <- cv.glmnet(x, y, family = "binomial")
plot(mushrooms_cvfit)

log(mushrooms_cvfit$lambda.min) 
# ������ ������ : �������� ���� ���� ���� lamda�� ���� ��������

log(mushrooms_cvfit$lambda.1se) 
# �ؼ� ������ ���� 

# lambda.1se �� �� �� �������� ����� ��� 
coef(mushrooms_cvfit,
     s = mushrooms_cvfit$lambda.1se)

# lambda.min �� �� �� �������� ����� ��� 
coef(mushrooms_cvfit,
     s = mushrooms_cvfit$lambda.min)


# �������� ��� ������ �߿� ������� �ִٰ� �Ǻ��� �������� ���� ��� 
length(which(coef(mushrooms_cvfit,
                  s="lambda.min")!=0))

length(which(coef(mushrooms_cvfit,
                  s="lambda.1se")!=0))


# Last Visualization ------------------

# ROC Curve, AUC Value
# TP FP TF FF ���̺��� ��� 
# ������ ����Ƚ�� ����غ���
# ����� ��� �켱���� ���� ������ ����غ��� 
# ���� ������Ʈ�� ��� ������ ������ ������ ������� ���� ������ �� �� ����. �̰͵� ����غ��� 


# ��� : ���� ���� �� ----------------
# �����°� ������ �� �� ���ϱ� 


# ��ũ�ٿ� �ۼ��ϱ� -----------








