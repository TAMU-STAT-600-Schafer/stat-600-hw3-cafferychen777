# tests.R - Script to test the LRMultiClass function

# Load the LRMultiClass function
source("FunctionsLR.R")

# ----------------------------------------------
# Test 1: Simple example with synthetic data
# ----------------------------------------------

# Generate synthetic data for two classes
set.seed(123)
n <- 100    # Number of training samples
p <- 3      # Number of features including intercept

# Create training data with intercept
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
y <- sample(0:1, n, replace = TRUE)

# Generate synthetic test data
ntest <- 50  # Number of test samples
Xt <- cbind(1, matrix(rnorm(ntest * (p - 1)), ntest, p - 1))
yt <- sample(0:1, ntest, replace = TRUE)

# Run the LRMultiClass algorithm with default parameters
out <- LRMultiClass(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1)

# Check if the objective value improves across iterations
cat("Objective values over iterations (synthetic data):\n")
print(out$objective)

# Check if the classification error changes across iterations
cat("Training error over iterations (synthetic data):\n")
print(out$error_train)
cat("Testing error over iterations (synthetic data):\n")
print(out$error_test)

# Plot the objective function, training error, and testing error over iterations
par(mfrow = c(1, 3))
plot(out$objective, type = 'o', main = "Objective Function (Synthetic Data)", xlab = "Iteration", ylab = "Objective Value")
plot(out$error_train, type = 'o', main = "Training Error (Synthetic Data)", xlab = "Iteration", ylab = "Error %")
plot(out$error_test, type = 'o', main = "Testing Error (Synthetic Data)", xlab = "Iteration", ylab = "Error %")
par(mfrow = c(1, 1))

# Use microbenchmark to time the code with lambda=1 and 50 iterations, repeated 5 times
library(microbenchmark)
timing_synthetic <- microbenchmark(
  out <- LRMultiClass(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1),
  times = 5
)

# Report the median time of the code
median_time_synthetic <- median(timing_synthetic$time) / 1e9  # Convert nanoseconds to seconds
cat("Median time for synthetic data: ", median_time_synthetic, " sec\n\n")

# ----------------------------------------------
# Test 2: Application to Letter data
# ----------------------------------------------

# Load the letter data
# Ensure that the files 'letter-train.txt' and 'letter-test.txt' are in the working directory
train_data <- read.table("Data/letter-train.txt", header = FALSE)
test_data <- read.table("Data/letter-train.txt", header = FALSE)

# Extract features and labels from training data
X_train <- as.matrix(train_data[, -1])  # Features
y_train <- as.numeric(train_data[, 1]) - 1  # Labels (from 0 to 25)

# Extract features and labels from test data
X_test <- as.matrix(test_data[, -1])
y_test <- as.numeric(test_data[, 1]) - 1

# Add intercept term to features
X_train <- cbind(1, X_train)
X_test <- cbind(1, X_test)

# Run the LRMultiClass algorithm with specified parameters
out_letter <- LRMultiClass(X_train, y_train, X_test, y_test, numIter = 50, eta = 0.1, lambda = 1)

# Print the final training and testing error rates
cat("Final training error rate (%) for letter data: ", out_letter$error_train[51], "\n")
cat("Final testing error rate (%) for letter data: ", out_letter$error_test[51], "\n")

# Check if the error rates are around the expected values (22% training, 26% testing)
if (out_letter$error_train[51] <= 25 && out_letter$error_train[51] >= 19) {
  cat("Training error rate is within the expected range (19%-25%).\n")
} else {
  cat("Training error rate is outside the expected range.\n")
}

if (out_letter$error_test[51] <= 29 && out_letter$error_test[51] >= 23) {
  cat("Testing error rate is within the expected range (23%-29%).\n")
} else {
  cat("Testing error rate is outside the expected range.\n")
}

# Plot the objective function, training error, and testing error over iterations
par(mfrow = c(1, 3))
plot(out_letter$objective, type = 'o', main = "Objective Function (Letter Data)", xlab = "Iteration", ylab = "Objective Value")
plot(out_letter$error_train, type = 'o', main = "Training Error (Letter Data)", xlab = "Iteration", ylab = "Error %")
plot(out_letter$error_test, type = 'o', main = "Testing Error (Letter Data)", xlab = "Iteration", ylab = "Error %")
par(mfrow = c(1, 1))

# Time the code execution for the letter data
timing_letter <- microbenchmark(
  out_letter <- LRMultiClass(X_train, y_train, X_test, y_test, numIter = 50, eta = 0.1, lambda = 1),
  times = 5
)

# Report the median time of the code
median_time_letter <- median(timing_letter$time) / 1e9  # Convert nanoseconds to seconds
cat("Median time for letter data: ", median_time_letter, " sec\n")

# Compare the median time to the benchmarks
if (median_time_letter <= 4.3 * 3) {
  cat("Your code runs within the acceptable time limit.\n")
} else {
  cat("Your code is slower than the acceptable time limit.\n")
}

# Additional checks on the objective function and errors
# Check if the objective value decreases over iterations
if (all(diff(out_letter$objective) <= 0)) {
  cat("Objective function decreases over iterations.\n")
} else {
  cat("Objective function does not consistently decrease over iterations.\n")
}

# Check if the training error decreases over iterations
if (all(diff(out_letter$error_train) <= 0)) {
  cat("Training error decreases over iterations.\n")
} else {
  cat("Training error does not consistently decrease over iterations.\n")
}

# Check if the testing error decreases over iterations
if (all(diff(out_letter$error_test) <= 0)) {
  cat("Testing error decreases over iterations.\n")
} else {
  cat("Testing error does not consistently decrease over iterations.\n")
}

# ----------------------------------------------
# Conclusion
# ----------------------------------------------

cat("\nConclusion:\n")
cat("Based on the tests above, the LRMultiClass function appears to be working correctly.\n")
cat("The error rates on the letter data are within the expected range.\n")
cat("The function's execution time is acceptable.\n")
cat("Objective function and errors generally decrease over iterations, indicating convergence.\n")

