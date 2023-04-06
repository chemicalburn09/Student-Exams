#Harvardx Capstone Project

#Working Directory
setwd("D:/R Workshop/Harvard Data Science/Student Exam project")

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(ggthemes)) install.packages("ggthemes")
if(!require(dplyr)) install.packages("dplyr")
if(!require(caTools)) install.packages("caTools")
if(!require(randomForest)) install.packages("randomForest")
if(!require(e1071)) install.packages("e1071")
if(!require(knitr)) install.packages("knitr")
if(!require(latexpdf)) install.packages("latexpdf")
if(!require(tinytex)) install.packages("tinytex")


#Load Libraries
library(tidyverse)
library(caret)
library(ggthemes)
library(dplyr)
library(caTools)
library(randomForest)
library(e1071)
library(knitr)
library(latexpdf)
library(tinytex)

#Load data
student_exams <- read.csv("https://raw.githubusercontent.com/chemicalburn09/Student-Exams/main/studentexams.csv")

#Data Pre-processing
#structure
str(student_exams)
head(student_exams)
dim(student_exams)

#blank/NA check
sum(is.na(student_exams))

#Convert character to factors
#student_exams$gender <- as.factor(student_exams$gender)
student_exams$ethnicity <- as.factor(student_exams$ethnicity)
student_exams$parental_edu <- as.factor(student_exams$parental_edu)
student_exams$lunch <- as.factor(student_exams$lunch)
student_exams$prep_course <- as.factor(student_exams$prep_course)

#Create new column for average score
student_exams$avg_score <- apply(student_exams[,6:8],1,mean)

#Summary of data
summary(student_exams)

#Exploratory Data Analysis #EDA

ggplot(student_exams, aes(x = gender)) + geom_bar()

ggplot(student_exams, aes(x = lunch)) + geom_bar()

##Student scores by gender
ggplot(student_exams, aes(x = gender, y = math_score, color = gender)) + geom_boxplot(outlier.colour="red") +
  stat_summary(fun = mean, geom="point", shape=23, size=4)

ggplot(student_exams, aes(x = gender, y = reading_score, color = gender)) + geom_boxplot(outlier.colour="red") +
  stat_summary(fun = mean, geom="point", shape=23, size=4)

ggplot(student_exams, aes(x = gender, y = writing_score, color = gender)) + geom_boxplot(outlier.colour="red") +
  stat_summary(fun = mean, geom="point", shape=23, size=4)

ggplot(student_exams, aes(x = (reading_score+math_score+writing_score/3), y = parental_edu, colour = gender)) +
  geom_point() +
  stat_summary(fun = mean, geom = "pointrange", size = 1.5)

ggplot(student_exams, aes(x = (reading_score+math_score+writing_score/3), y = ethnicity, colour = gender)) +
  geom_point() +
  stat_summary(fun = mean, geom = "pointrange", size = 1.5)

#Binary coding ?Do we need it? prior to model build...
#student_exams$gender <- ifelse(student_exams$gender == "male", 1, 0) # Male = 1, Female = 0
#student_exams$lunch <- ifelse(student_exams$lunch == "standard", 1, 0) # Standard = 1, Free/Reduced = 0
#student_exams$prep_course <- ifelse(student_exams$prep_course == "completed", 1, 0) # Yes = 1, No = 0
#student_exams$lunch <- as.numeric(student_exams$lunch)
#student_exams$gender <- as.numeric(student_exams$gender)
#student_exams$prep_course <- as.numeric(student_exams$prep_course)
#student_exams$math_score <- as.numeric(student_exams$math_score)
#student_exams$reading_score <- as.numeric(student_exams$reading_score)
#student_exams$writing_score <- as.numeric(student_exams$writing_score)

## Convert ethnicity to numeric (Group A = 1, Group B = 2, Group C = 3, 
## Group D = 4, Group E = 5)
##student_exams$ethnicity <- as.numeric(student_exams$ethnicity)

## Convert ethnicity to numeric (associate's degree = 1, bachlor's degree = 2,
## high school = 3, master's degree = 4, some college = 5, some high school = 6)
##student_exams$parental_edu <- as.numeric(student_exams$parental_edu)

#Partition Data and set seed
##splitting student_exams data into test & training 80/20 split

set.seed(1984)

split = sample.split(student_exams, SplitRatio = 0.80)
student_train = subset(student_exams, split == TRUE)
student_test = subset(student_exams, split == FALSE)

#Scale data (scores)
student_train$math_score <- scale(student_train$math_score)
student_train$writing_score <- scale(student_train$writing_score)
student_train$reading_score <- scale(student_train$reading_score)
student_train$avg_score <- scale(student_train$avg_score)

student_test$math_score <- scale(student_test$math_score)
student_test$writing_score <- scale(student_test$writing_score)
student_test$reading_score <- scale(student_test$reading_score)
student_test$avg_score <- scale(student_test$avg_score)

#Model Building #1 multivariate linear regression math scores
set.seed(1984)

model10 = lm(math_score ~ reading_score + writing_score, data = student_train)
summary(model10)
varImp(model10)

y_pred10 <- predict(model10, student_test)
#summary(y_pred)

RMSE(y_pred10, student_test$math_score)
##Returned RMSE scaled is 0.5598695

#Model Building #2 multivariate linear regression reading scores
set.seed(1984)

model11 = lm(reading_score ~ math_score + writing_score, data = student_train)
summary(model11)
varImp(model11)

y_pred11 <- predict(model11, student_test)
#summary(y_pred)

RMSE(y_pred11, student_test$math_score)
##Returned RMSE scaled is 0.536046

#MOdel Building #3 multivariate linear regression writing scores
set.seed(1984)

model12 = lm(writing_score ~ math_score + reading_score, data = student_train)
summary(model12)
varImp(model12)

y_pred12 <- predict(model12, student_test)
#summary(y_pred)

RMSE(y_pred12, student_test$math_score)
##Returned RMSE scaled is 0.5214318


#Model Building #4 Randomforest
#remove average column so that it doesn't skew results

set.seed(1984)

classifier = randomForest(x = student_train[,-9],
                          y = student_train$math_score,
                          ntree = 500, random_state = 0)

y_pred13 = predict(classifier, student_test[,-9])

RMSE(y_pred13, student_test$math_score)
##Returned RMSE scaled is 0.1114175

varImp(classifier)
