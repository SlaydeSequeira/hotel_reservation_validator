library(reticulate)
library(parallel)

use_python("/path/to/python", required = TRUE)

# Import Python packages
pd <- import("pandas")
sklearn <- import("sklearn")
pickle <- import("pickle")

# Load data
data <- pd$read_csv("Hotel Reservations.csv")

# Drop non-essential columns
data <- data$drop(columns = "Booking_ID")

# Encode categorical variables using MapReduce parallel processing
LabelEncoder <- sklearn$preprocessing$LabelEncoder

# Define function to encode a single column (Map step)
encode_column <- function(column_name) {
  le <- LabelEncoder()
  encoded <- le$fit_transform(data[[column_name]])
  return(list(column = column_name, encoded = encoded, encoder = le))
}

# Columns to encode
categorical_columns <- c("type_of_meal_plan", "room_type_reserved", "market_segment_type", "booking_status")

# Map step: Parallel encoding
label_encoders <- mclapply(categorical_columns, encode_column, mc.cores = detectCores())

# Reduce step: Assign encoded data back to the data frame
data <- Reduce(function(df, result) {
  df[[result$column]] <<- result$encoded
  return(df)
}, label_encoders, init = data)

# Convert list of encoders to a named list for easy access
label_encoders <- setNames(lapply(label_encoders, function(x) x$encoder), categorical_columns)

# Define features and target
X <- data$drop(columns = "booking_status")
y <- data[["booking_status"]]

# Split data
train_test_split <- sklearn$model_selection$train_test_split
X_train, X_test, y_train, y_test <- train_test_split(X, y, test_size = 0.2, random_state = 42)

# Standardize features using MapReduce parallel processing
StandardScaler <- sklearn$preprocessing$StandardScaler

# Map step: Fit and transform training data, transform test data
scaler <- StandardScaler()
X_train <- mclapply(X_train, function(column) scaler$fit_transform(matrix(column, ncol = 1)), mc.cores = detectCores())
X_test <- mclapply(X_test, function(column) scaler$transform(matrix(column, ncol = 1)), mc.cores = detectCores())

# Reduce step: Convert the processed list back to matrix form
X_train <- do.call(cbind, X_train)
X_test <- do.call(cbind, X_test)

# Train Logistic Regression model
LogisticRegression <- sklearn$linear_model$LogisticRegression
model <- LogisticRegression(max_iter = 1000, random_state = 42)
model$fit(X_train, y_train)

# Save model, scaler, and label encoders to a .pkl file
model_data <- list(model = model, scaler = scaler, label_encoders = label_encoders)

output_file <- "logistic_regression_model.pkl"
with(open(output_file, "wb") %as% file, {
  pickle$dump(model_data, file)
})

cat("Model saved to", output_file)
