setwd("D:\\Course\\R\\Human Resource")

ld_train = read.csv("bank-full_train.csv", stringsAsFactors = F)
ld_test = read.csv("bank-full_test.csv", stringsAsFactors = F)

library(dplyr)
library(tidyr)
library(visdat)
library(randomForest)
library(randomForestSRC)
library(ggplot2)
library(pROC)
library(car)
library(lubridate)

View(ld_train)

ld_test$y = NA

ld_train$data = "train"
ld_test$data = "test"

ld_all = rbind(ld_train,ld_test)


lapply(ld_all, function(x) sum(is.na(x)))

table(ld_all$age)


ld_all = ld_all %>% 
  mutate(age_btw_18_30 = as.numeric(age <= 30),
         age_btw_31_40 = as.numeric(age>30 & age<=40),
         age_btw_41_50 = as.numeric(age>40 & age<=50),
         age_older_than_50 = as.numeric(age>50)) %>% 
  select(-age)


convert_to_numeric= c("default","housing","loan")

for(i in convert_to_numeric){
  ld_all[,i] = as.numeric(ld_all[,i]=="yes")
}

ld_all$contact = NULL

glimpse(ld_all)


ld_all$month[ld_all$month=="jan"]=1
ld_all$month[ld_all$month=="feb"]=2
ld_all$month[ld_all$month=="mar"]=3
ld_all$month[ld_all$month=="apr"]=4
ld_all$month[ld_all$month=="may"]=5
ld_all$month[ld_all$month=="jun"]=6
ld_all$month[ld_all$month=="jul"]=7
ld_all$month[ld_all$month=="aug"]=8
ld_all$month[ld_all$month=="sep"]=9
ld_all$month[ld_all$month=="oct"]=10
ld_all$month[ld_all$month=="nov"]=11
ld_all$month[ld_all$month=="dec"]=12


ld_all$last_contacted_date = paste(ld_all$day,ld_all$month, sep = "-")

ld_all$last_contacted_date = as.Date(ld_all$last_contacted_date, format = "%d-%m")

ld_all$duration = NULL


table(ld_all$previous)
ld_all$pdays[ld_all$pdays<0]=0
ld_all$pdays[ld_all$pdays>0]=1

ld_all$previous[ld_all$previous>0]=1

ld_all$ID = NULL
ld_all$day = NULL
ld_all$month = NULL


CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}

dummies = c("job","marital","education","poutcome")


for(i in dummies){
  ld_all = CreateDummies(ld_all, i , 10)
}

ld_all$y = as.numeric(ld_all$y == "yes")

ld_train=ld_all %>% filter(data=='train') %>% select(-data)
ld_test=ld_all %>% filter(data=='test') %>% select(-data,-y)


set.seed(2)
s=sample(1:nrow(ld_train),0.8*nrow(ld_train))
lgr_train1=ld_train[s,]
lgr_train2=ld_train[-s,]

for_vif=lm(y~.,data = lgr_train1)


log_fit = glm(y~., data= lgr_train1)

log_fit = step(log_fit)
formula(log_fit)
summary(log_fit)

log_fit = glm(y ~ default + balance + housing + loan + campaign + age_btw_18_30 +
                age_btw_31_40 + age_btw_41_50 + job_student + job_housemaid +
                job_unemployed + job_retired + job_admin. + job_technician +
                job_management + marital_married + education_primary + education_tertiary +
                poutcome_other + poutcome_failure + poutcome_unknown, data= lgr_train1, family = 'binomial')


summary(log_fit)


val.score=predict(log_fit,newdata = lgr_train2, type='response')


auc(roc(lgr_train2$y,val.score))

log_fit_final= glm(y ~ default + balance + housing + loan + campaign + age_btw_18_30 +
                     age_btw_31_40 + age_btw_41_50 + job_student + job_housemaid +
                     job_unemployed + job_retired + job_admin. + job_technician +
                     job_management + marital_married + education_primary + education_tertiary +
                     poutcome_other + poutcome_failure + poutcome_unknown, data= ld_train, family = 'binomial')

summary(log_fit_final)


test_prob_score = predict(log_fit_final, newdata = ld_test, type='response',row.names= F )


train.score=predict(log_fit_final,newdata = ld_train,type='response')

real=ld_train$y

cutoffs=seq(0.001,0.999,0.001)

cutoff_data=data.frame(cutoff=99999,Sn=99999,Sp=99999,KS=9999,F5=9999,F.1=9999,M=9999)

for(cutoff in cutoffs){
  
  ## Conversion into hard calsses
  predicted=as.numeric(train.score>cutoff)
  
  
  TP=sum(real==1 & predicted==1)
  TN=sum(real==0 & predicted==0)
  FP=sum(real==0 & predicted==1)
  FN=sum(real==1 & predicted==0)
  
  P=TP+FN
  N=TN+FP
  
  Sn=TP/P
  Sp=TN/N
  precision=TP/(TP+FP)
  recall=Sn
  KS=(TP/P)-(FP/N)
  F5=(26*precision*recall)/((25*precision)+recall)
  F.1=(1.01*precision*recall)/((.01*precision)+recall)
  
  M=(100*FP+TP)/(5*(P+N))
  
  cutoff_data=rbind(cutoff_data,c(cutoff,Sn,Sp,KS,F5,F.1,M))
}

cutoff_data=cutoff_data[-1,]

View(cutoff_data)

final_cutoff = 0.136

test_prob_score[test_prob_score>=0.136]="Yes"
test_prob_score[test_prob_score<0.136]="No"

table(test_prob_score)

write.csv(test_prob_score,"Atahar_Budihal_P5_part2.csv", row.names = F)



































