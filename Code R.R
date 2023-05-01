
install.packages("dplyr")
install.packages("glmnet")
install.packages("ggplot2")
install.packages("caret")


### LASSO regression ###

# On charge la librairie 

library(glmnet)
set.seed(123)
# define predictor and response variables

y <- longley$Employed
x <- data.matrix(longley[, c('GNP.deflator', 'GNP', 'Unemployed', 'Armed.Forces','Population','Year')])

# fit ridge regression model

modelLasso <- glmnet(x, y, alpha = 1)
summary(modelLasso)

# perform k-fold cross-validation to find optimal lambda value

cvLasso_model <- cv.glmnet(x, y, alpha = 1)
plot(cvLasso_model)

# find optimal lambda value that minimizes test MSE

bestLasso_lambda <- cvLasso_model$lambda.min
bestLasso_lambda

# produce Ridge trace plot

plot(modelLasso, xvar = "lambda")

# find coefficients of best model

bestLasso_model <- glmnet(x, y, alpha = 1, lambda = bestLasso_lambda)
coef(bestLasso_model)

# calculate R-squared of model on training data

y_predictedLasso <- predict(modelLasso, s = bestLasso_lambda, newx = x)

# find SST and SSE

sstLasso <- sum((y - mean(y))^2)
sseLAasso <- sum((y_predictedLasso - y)^2)

# find R-Squared

rsqLasso <- 1 - sseLAasso / sstLasso
rsqLasso



### RIDGE regression ###

# On charge les librairies 

   
library(dplyr)   
library(psych)   

library(glmnet)
set.seed(123)
# define predictor and response variables

y <- longley$Employed
x <- data.matrix(longley[, c('GNP.deflator', 'GNP', 'Unemployed', 'Armed.Forces','Population','Year')])

# fit ridge regression model

modelRidge <- glmnet(x, y, alpha = 0)
summary(modelRidge)

# perform k-fold cross-validation to find optimal lambda value

cvRidge_model <- cv.glmnet(x, y, alpha = 0)
plot(cvRidge_model)

# find optimal lambda value that minimizes test MSE

bestRidge_lambda <- cvRidge_model$lambda.min
bestRidge_lambda

# produce Ridge trace plot

plot(modelRidge, xvar = "lambda")

# find coefficients of best model

bestRidge_model <- glmnet(x, y, alpha = 0, lambda = bestRidge_lambda)
coef(bestRidge_model)

# calculate R-squared of model on training data

y_predictedRidge <- predict(modelRidge, s = bestRidge_lambda, newx = x)

# find SST and SSE

sstRidge <- sum((y - mean(y))^2)
sseRidge <- sum((y_predictedRidge - y)^2)

# find R-Squared

rsqRidge <- 1 - sseRidge / sstRidge
rsqRidge



# same application with elastic net 


library(ggplot2)
library(caret)

set.seed(123)
data("longley")

x <- longley %>% 
  select(Employed) %>% 
  scale(center = TRUE, scale = FALSE) %>% 
  as.matrix()
y <- longley %>% 
  select(-Employed) %>% 
  as.matrix()

# Set training control

train_control <- trainControl(method = "repeatedcv",
                              number = 12,
                              repeats = 12,
                              search = "random",
                              verboseIter = TRUE)

# Train the model

elastic_net_model <- train(Employed ~ .,
                           data = cbind(y, x),
                           method = "glmnet",
                           preProcess = c("center", "scale"),
                           tuneLength = 35,
                           trControl = train_control)

# on affiche le meilleur lambda et le meilleur alpha 

elastic_net_model$bestTune


#  Model Prediction

x_hat_pre <- predict(elastic_net_model, y)
x_hat_pre

# Multiple R-squared

rsqElasticNet <- cor(x, x_hat_pre)^2
rsqElasticNet

# Plot

plot(elastic_net_model, main = "Elastic Net Regression")

