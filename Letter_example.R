# Application of multi-class logistic to letters data

# Load the letter data
#########################
# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])
X <- cbind(1, X)  # Add intercept column

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])
Xt <- cbind(1, Xt)  # Add intercept column

# Source the LR function
source("FunctionsLR.R")

# Run the LRMultiClass algorithm with lambda = 1 and 50 iterations
out <- LRMultiClass(X, Y, Xt, Yt, lambda = 1, numIter = 50)

# Plot the objective function, training error, and testing error over iterations
plot(out$objective, type = 'o', main = "Objective Function", xlab = "Iteration", ylab = "Objective Value")
plot(out$error_train, type = 'o', main = "Training Error", xlab = "Iteration", ylab = "Error %")
plot(out$error_test, type = 'o', main = "Testing Error", xlab = "Iteration", ylab = "Error %")

# Use microbenchmark to time the code with lambda=1 and 50 iterations, repeated 5 times
library(microbenchmark)
timing <- microbenchmark(out <- LRMultiClass(X, Y, Xt, Yt, lambda = 1, numIter = 50), times = 5)

# Report the median time of the code
median_time <- median(timing$time) / 1e9  # Convert to seconds
cat("Median time: ", median_time, " sec\n")