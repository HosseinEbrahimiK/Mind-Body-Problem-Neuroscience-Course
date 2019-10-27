#data_read
d <- read.csv("/Users/joliaserm/Desktop/data_neuro.csv", header = TRUE)

#ttest
true_false <- (d$Response == d$Stimulus)
true_false <- as.integer(true_false)
num_of_true <- sum(true_false)
num_of_false <- length(true_false) - num_of_true
sample_size <- length(true_false)

test <- prop.test(num_of_true, sample_size, alternative = "greater")