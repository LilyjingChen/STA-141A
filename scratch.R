rm(list=ls())
library(caret)
require(fastDummies)
library(tidyr)
library(dplyr)

# Import data
dataset<- read.csv("bank-additional.csv", sep=";")

# Remove the unknowns 
dataset <- dataset[dataset$education != 'unknown' & dataset$default != "unknown"
                   & dataset$job != "unknown" & dataset$marital != "unknown" 
                   & dataset$housing != "unknown" & dataset$loan != "unknow",]

# one hot encoding for 'education' and 'marital' 
clean_data <- dummy_cols(dataset,select_columns = c("marital","job","education"))
# Remove the original 'education' and 'marital' column
clean_data$marital <- NULL
clean_data$education <- NULL
clean_data$job <- NULL

# Create dummy for 'default' 1(yes);0(no)
clean_data <- clean_data %>% mutate(default.dum = ifelse(as.character(default)== 'yes',1,0))
# Remove the original 'default' column
clean_data$default <- NULL

# Create dummy for 'housing' 1(yes);0(no)
clean_data <- clean_data %>% mutate(housing.dum = ifelse(as.character(housing)== 'yes',1,0))
# Remove the original 'housing' column
clean_data$housing <- NULL

# Create dummy for 'loan' 1(yes);0(no)
clean_data <- clean_data %>% mutate(loan.dum = ifelse(as.character(loan)== 'yes',1,0)) 
# Remove the original 'loan' column
clean_data$loan <- NULL

# Create dummy for 'contact' 1(telephone);0(cellular)
clean_data <- clean_data %>% mutate(contact.dum = ifelse(as.character(contact)== 'telephone',1,0)) 
# Remove the original 'contact' column
clean_data$contact <- NULL

# y dummy(subscribed a term deposit) 1(yes);0(otherwise)
clean_data <- clean_data%>% mutate(y.dum = ifelse(as.character(y)== 'yes',1,0))
# remove the original 'y' column
clean_data$y <- NULL

# poutcome dummy (previous marketing campaign result) 1(success);0(failure or nonexistent)
clean_data <- clean_data%>% mutate(poutcome.dum = ifelse(as.character(poutcome)== 'success',1,0))
# Remove the original 'poutcome' column
clean_data$poutcome <- NULL

# 'duration' is not useful for predictive purpose because it has strong relationship with the output variable (y)
clean_data$duration <- NULL

# Without month & day_of_week 
# The two columns (last contact month of year) and (last contact day of the week) are not useful for our prediction, so we eliminate them in our data
cl_data <- clean_data
cl_data$month <- NULL
cl_data$day_of_week <- NULL

# One hot encoding for 'education' and 'marital' 
#clean_data <- dummy_cols(clean_data,select_columns = c("month","day_of_week"))
# Remove the original 'month' and 'day_of_week' column
#$month <- NULL
#clean_data$day_of_week <- NULL

#numeric 
clean_data$month <- as.numeric(as.factor(clean_data$month))
clean_data$day_of_week <- as.numeric(as.factor(clean_data$day_of_week))
clean_data$month <- NULL
clean_data$day_of_week <- NULL

head(clean_data)

## NN
require(neuralnet)
smp_size <- floor(0.7 * nrow(clean_data))
set.seed(1)
train_ind <- sample(seq_len(nrow(clean_data)), size = smp_size)
train <- clean_data[train_ind, ]
train_y <- train$y.dum
#train$y.dum <- NULL
test <-clean_data[-train_ind,]
test_y <- test$y.dum
#test$y.dum <- NULL

names(train)[names(train) == "job_blue-collar"] <- "job_blue_collar"
names(train)[names(train) == "job_self-employed"] <- "job_self_employed"

names(test)[names(test) == "job_blue-collar"] <- "job_blue_collar"

names(test)[names(test) == "job_self-employed"] <- "job_self_employed"



nn <- neuralnet(y.dum ~age+campaign+pdays+previous+emp.var.rate+cons.price.idx+cons.conf.idx+euribor3m+nr.employed
                +marital_divorced+marital_married+marital_single+job_admin.+`job_blue_collar`+job_entrepreneur+job_housemaid
                +job_management+job_retired+job_self_employed+job_services+job_student+job_technician+job_unemployed
                +education_basic.4y+education_basic.6y+education_basic.9y+education_high.school+education_illiterate
                +education_professional.course+education_university.degree+default.dum+housing.dum+loan.dum+contact.dum+poutcome.dum,
                data=train, hidden=c(2,2),act.fct = "logistic", err.fct = "sse",
                linear.output = FALSE, lifesign = "full",rep = 100 )
plot(nn,rep='best',col.hidden = 'darkgreen',
     col.hidden.synapse = 'darkgreen',
     show.weights = F,
     information = F,
     fill = 'lightblue')
summary(nn)
### Training 
set.seed(1)
output1<-compute(nn, train)
cl1 <- output1$net.result
pred1 <- ifelse(cl1>0.5, 1, 0)
(tab1 <- table(pred1, train$y.dum))
(miscal_err1 <- 1-sum(diag(tab1))/sum(tab1))

## Testing 
output2<-compute(nn, test)
cl2 <- output2$net.result
pred2 <- ifelse(cl2>0.5, 1, 0)
tab2 <- table(pred2, test$y.dum)
miscal_err2 <- 1-sum(diag(tab2))/sum(tab2)

names(train)[names(train) == "job_blue-collar"] <- "job_blue_collar"
names(train)[names(train) == "job_self-employed"] <- "job_self_employed"

names(test)[names(test) == "job_blue-collar"] <- "job_blue_collar"

names(test)[names(test) == "job_self-employed"] <- "job_self_employed"


train$`job_blue-collar`
## Using Keras 
require(keras)
n <- neuralnet(y.dum ~age+campaign+pdays+previous+emp.var.rate+cons.price.idx+cons.conf.idx+euribor3m+nr.employed
               +marital_divorced+marital_married+marital_single+job_admin.+`job_blue_collar`+job_entrepreneur+job_housemaid
               +job_management+job_retired+job_self_employed+job_services+job_student+job_technician+job_unemployed
               +education_basic.4y+education_basic.6y+education_basic.9y+education_high.school+education_illiterate
               +education_professional.course+education_university.degree+default.dum+housing.dum+loan.dum+contact.dum+poutcome.dum,
               data=train, hidden=c(10,6),act.fct = "logistic", err.fct = "sse",
               linear.output = FALSE, lifesign = "minimal",rep = 10)
plot(n, col.hidden = 'darkgreen',
     col.hidden.synapse = 'darkgreen',
     show.weights = F,
     information = F,
     fill = 'lightblue',rep = 'best')

mean <- colMeans(train)
s <- apply(train, 2, sd)
train <- scale(train , center = mean, scale = s)
test <- scale(test, center = mean, scale = s)

train_x <- data.frame(train[,1:34],train[,'poutcome.dum'])
test_x <- data.frame(test[,1:34],test[,'poutcome.dum'])


model<-keras_model_sequential()
model%>%layer_dense(units = 3, activation = 'sigmoid', input_shape = 35 )%>%layer_dense(units = 1)

# compile 
model%>%compile(loss='mse', optimize = 'sgd', metrics = 'accuracy')

# Fit model 
train <- clean_data[train_ind, ]
train_y <- train[,35]
trainy <- data.matrix(train_y)
train_y <- train[,35]
train$y.dum <- NULL

#train_y <- to_categorical(train_y, 2)
mymodel <- model%>%fit(train, trainy, epochs = 100, batch_size = 20,validation_split = 0.3)
                      
