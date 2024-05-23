library(ggplot2)

# In this work we apply t-tests to look for the best model for LUNG CANCER CLASSIFICATION, we have at our disposal two models, trained with a grid of hyperparameters
# 1) We perform one-sample t-test on the accuracy of two models separately, which is normally distributed, to check whether the accuracy of the models is different from 80%
# 2) We perform two-sample t-test on the accuracy of the two models, to check whether they perform differently or not (in terms of accuracy)
# 3) We perform Non-parametric two-sample test on the loss, which is not normally distributed
# 4) We perform One-way ANOVA test on the accuracy, wrt the learning rate, to check whether the learning rate affects or not the accuracy
# 5) We perform Two-way ANOVA test on the accuracy, wrt the learning rate and weight decay, to check whether the learning rate and weight decay affects or not the accuracy
# CONCLUSION: The model isn't affected by any of the chosen parameters


dataset = read.csv("D:\\Statistica Applicata\\DL hyperparameters study - lung cancer detection\\hyperparameters_study.csv")

dataset$dropout = as.factor(dataset$dropout)
dataset$weight_decay = as.factor(dataset$weight_decay)
dataset$lr = as.factor(dataset$lr)
dataset$hidden_size = as.factor(dataset$hidden_size)
dataset <- dataset[, -which(names(dataset) == "X")]

summary(dataset)

# ----------------------------------------- 1) ONE-SAMPLE TESTS ---------------------------------------

# CHECK NORMALITY ASSUMPTION OF ACCURACY.
# Normality assumption holds somewhat, but the sample size is very small. We can still use t-test though.

par(mfrow=c(2,2))
hist(dataset$accuracy[which(dataset$dropout == "False")])
qqnorm(dataset$accuracy[which(dataset$dropout == "False")])
qqline(dataset$accuracy[which(dataset$dropout == "False")],col="red",lwd=2)

hist(dataset$accuracy[which(dataset$dropout == "True")])
qqnorm(dataset$accuracy[which(dataset$dropout == "True")])
qqline(dataset$accuracy[which(dataset$dropout == "True")],col="red",lwd=2)


# ONE-SAMPLE HYPOTHESIS TESTING: test that the accuracy is different from 80%
# * Base hypothesis states that the accuracy is 80%
# * Alternative hypothesis states that accuracy is different from 80%

# p-value is less that 0.05 on the first model, hence the accuracy of the model is different from 80%
t.test(x = dataset$accuracy[which(dataset$dropout == "False")], mu = 0.8)
# p-value is less that 0.05 on the second model, hence the accuracy of the model is different from 80%
t.test(x = dataset$accuracy[which(dataset$dropout == "True")], mu = 0.8)


# ----------------------------------------- 2) TWO-SAMPLE TESTS ---------------------------------------

# TWO-SAMPLE HYPOTHESIS TESTING: test whether one model is better than the other
# * Base hypothesis states that the difference between the true mean accuracy of the two models is zero (the models perform the same)
# * Alternative hypothesis states that the difference between the true mean accuracy of the two models is different from zero (the models perform differently)

# p-value is greater than 0.05, hence the models perform the same.
t.test(x = dataset$accuracy[which(dataset$dropout == "False")], y = dataset$accuracy[which(dataset$dropout == "True")], mu = 0)

# ----------------------------------------- 3) NON-PARAMETRIC TESTS ---------------------------------------
# We accept the base hypothesis that the two models have the same loss.

par(mfrow=c(1,2))
qqnorm(dataset$loss[which(dataset$dropout == "False")])
qqline(dataset$loss[which(dataset$dropout == "False")], lwd=2)
qqnorm(dataset$loss[which(dataset$dropout == "True")])
qqline(dataset$loss[which(dataset$dropout == "True")], lwd=2)

wilcox.test(x = dataset$loss[which(dataset$dropout == "False")], y = dataset$loss[which(dataset$dropout == "True")], alternative = "two.sided")


# ----------------------------------------- 4) ONE-WAY ANOVA TEST -----------------------------------------------------
# The Base assumptions is accepted, hence changing the learning rate DOESN'T AFFECT accuracy for the model with dropout

model_dropout = dataset[which(dataset$dropout == "True"),]

# CHECK ASSUMPTIONS
# Check that each category is normally distributed
par(mfrow=c(1,3))
qqnorm(model_dropout$loss[which(model_dropout$lr == 1e-3)])
qqline(model_dropout$loss[which(model_dropout$lr == 1e-3)], lwd=2)
qqnorm(model_dropout$loss[which(model_dropout$lr == 1e-4)])
qqline(model_dropout$loss[which(model_dropout$lr == 1e-4)], lwd=2)
qqnorm(model_dropout$loss[which(model_dropout$lr == 1e-5)])
qqline(model_dropout$loss[which(model_dropout$lr == 1e-5)], lwd=2)

# Check that each category has more or less the same variance
var(model_dropout$loss[which(model_dropout$lr == 1e-3)])
var(model_dropout$loss[which(model_dropout$lr == 1e-4)])
var(model_dropout$loss[which(model_dropout$lr == 1e-5)])


# ASSUMPTIONS CHECKED: We can use ANova
one.way = aov(accuracy ~ lr, data = model_dropout)
summary(one.way)

# ----------------------------------------- 5) TWO-WAY ANOVA TEST -----------------------------------------------------
# Accuracy isn't affected NEITHER BY THE LEARNING RATE, NOR BY WEIGHT DECAY

two.way = aov(accuracy ~ weight_decay + lr, data = model_dropout)
summary(two.way)
