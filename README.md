---
title: "Build and deploy a stroke prediction model using R"
author: "Par Bah"
date: "`r Sys.Date()`"
output:
  html_document: default
  pdf_document:
    latex_engine: xelatex
---


# About Data Analysis Report

This RMarkdown file contains the report of the data analysis done for the project on building and deploying a stroke prediction model in R. It contains analysis such as data exploration, summary statistics and building the prediction models. The final report was completed on `r date()`. 

**Data Description:**

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.

This data set is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relevant information about the patient.


# Task One: Import data and data preprocessing

## Load data and install packages

```{r}
# Install required packages (only run once)
install.packages("tidyverse")
install.packages("readr")

# Load libraries
library(tidyverse)
library(readr)

# Load dataset (with file path included)
stroke_data <- read_csv("healthcare-dataset-stroke-data.csv")

# Quick look at the dataset
glimpse(stroke_data)
head(stroke_data)


```


## Describe and explore the data

```{r}
# =========================
# Describe & Explore the Data (creative color palette)
# =========================

# ---- 0) Packages ----
req_pkgs <- c("tidyverse", "viridis", "RColorBrewer")
new_pkgs <- req_pkgs[!(req_pkgs %in% installed.packages()[,"Package"])]
if (length(new_pkgs)) install.packages(new_pkgs, quiet = TRUE)
library(tidyverse)
library(viridis)
library(RColorBrewer)

theme_set(theme_minimal(base_size = 13))

# ---- 1) Load data ----
data_path <- "healthcare-dataset-stroke-data.csv"  # edit path if needed
stroke_data_raw <- read_csv(data_path, show_col_types = FALSE)

# ---- 2) Type-safe preprocessing ----
stroke_data <- stroke_data_raw %>%
  mutate(
    bmi = na_if(bmi, "N/A"),
    bmi = na_if(bmi, "Unknown"),
    bmi = suppressWarnings(as.numeric(bmi)),
    age = suppressWarnings(as.numeric(age)),
    avg_glucose_level = suppressWarnings(as.numeric(avg_glucose_level))
  ) %>%
  mutate(
    across(any_of(c("gender","ever_married","work_type","Residence_type","smoking_status")),
           ~ as.factor(.)),
    across(any_of(c("hypertension","heart_disease","stroke")), ~ as.factor(.))
  )
if ("id" %in% names(stroke_data)) stroke_data <- select(stroke_data, -id)

# ---- 3) Quick overview ----
glimpse(stroke_data)
colSums(is.na(stroke_data))

# ---- 4) Class distribution ----
if ("stroke" %in% names(stroke_data)) {
  class_dist <- stroke_data %>%
    count(stroke) %>%
    mutate(percentage = round(100 * n / sum(n), 2))
  
  ggplot(class_dist, aes(x = stroke, y = n, fill = stroke)) +
    geom_col(width = 0.6, color = "black", alpha = 0.9) +
    scale_fill_viridis(discrete = TRUE, option = "plasma") +
    labs(title = "Class Distribution: Stroke", x = "Stroke", y = "Count") +
    geom_text(aes(label = paste0(percentage, "%")), vjust = -0.5, size = 5) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
}

# ---- 5) Categorical distributions ----
cat_vars <- intersect(c("gender","ever_married","work_type","Residence_type","smoking_status"),
                      names(stroke_data))

for (cv in cat_vars) {
  p <- stroke_data %>%
    count(.data[[cv]]) %>%
    ggplot(aes(x = reorder(.data[[cv]], n), y = n, fill = .data[[cv]])) +
    geom_col(color = "white", width = 0.7, alpha = 0.85) +
    coord_flip() +
    scale_fill_brewer(palette = "PuBuGn") +
    labs(title = paste("Distribution of", cv), x = cv, y = "Count") +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold", hjust = 0.5))
  print(p)
}

# ---- 6) Numeric distributions ----
numeric_vars <- names(select(stroke_data, where(is.numeric)))
if (length(numeric_vars) > 0) {
  stroke_data %>%
    select(all_of(numeric_vars)) %>%
    pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
    ggplot(aes(x = value, fill = variable)) +
    geom_histogram(bins = 30, color = "white", alpha = 0.8) +
    facet_wrap(~ variable, scales = "free", ncol = 2) +
    scale_fill_viridis(discrete = TRUE, option = "magma") +
    labs(title = "Distribution of Numeric Variables") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          legend.position = "none")
}

# ---- 7) Stroke relationships (categorical) ----
if ("stroke" %in% names(stroke_data)) {
  for (cv in cat_vars) {
    p2 <- stroke_data %>%
      count(.data[[cv]], stroke) %>%
      ggplot(aes(x = .data[[cv]], y = n, fill = stroke)) +
      geom_col(position = "dodge", color = "white", alpha = 0.9) +
      scale_fill_viridis(discrete = TRUE, option = "inferno") +
      labs(title = paste("Stroke by", cv), x = cv, y = "Count") +
      theme(axis.text.x = element_text(angle = 30, hjust = 1),
            plot.title = element_text(face = "bold", hjust = 0.5))
    print(p2)
  }
}

# ---- 8) Stroke relationships (numeric) ----
if ("stroke" %in% names(stroke_data) & length(numeric_vars) > 0) {
  for (nv in numeric_vars) {
    p3 <- ggplot(stroke_data, aes(x = stroke, y = .data[[nv]], fill = stroke)) +
      geom_boxplot(alpha = 0.7, outlier.colour = "#FF6F61", outlier.alpha = 0.6) +
      scale_fill_brewer(palette = "PuRd") +
      labs(title = paste(nv, "by Stroke"), x = "Stroke", y = nv) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5),
            legend.position = "none")
    print(p3)
  }
}

# =========================
# End: Describe & Explore
# =========================


```



# Task Two: Build prediction models

```{r}
# =========================
# Task Two: Build Prediction Models (repaired)
# =========================

# ---- 0) Packages ----
req_pkgs <- c("caret", "randomForest", "rpart", "pROC")
new_pkgs <- req_pkgs[!(req_pkgs %in% installed.packages()[,"Package"])]
if (length(new_pkgs)) install.packages(new_pkgs, quiet = TRUE)

library(caret)
library(randomForest)
library(rpart)
library(pROC)

set.seed(123)

# ---- 1) Start from your cleaned EDA dataset `stroke_data` ----
# (If you still have 'id', drop it)
if ("id" %in% names(stroke_data)) stroke_data <- subset(stroke_data, select = -id)

# Ensure all character categoricals are factors
stroke_data <- stroke_data %>%
  mutate(across(where(is.character), as.factor))

# Simple numeric impute (median) to avoid losing rows with NAs
stroke_data <- stroke_data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# ---- 2) FIX: target labels must be valid variable names ("No","Yes") ----
stroke_data$stroke <- factor(
  ifelse(as.character(stroke_data$stroke) %in% c("1","Yes","Stroke","TRUE"), "Yes", "No"),
  levels = c("No","Yes")
)

# ---- 3) Train/test split ----
train_index <- createDataPartition(stroke_data$stroke, p = 0.8, list = FALSE)
train_data  <- stroke_data[train_index, ]
test_data   <- stroke_data[-train_index, ]

# ---- 4) Train Control (ROC with upsampling for class imbalance) ----
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,  # gives ROC/Se/Sp
  sampling = "up",                    # handle imbalance (upsample minority in each resample)
  savePredictions = "final"
)

# Common recipe: center/scale numeric predictors
pp <- c("center","scale")

# ---- 5) Logistic Regression ----
set.seed(123)
model_log <- train(
  stroke ~ ., data = train_data,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  preProcess = pp,
  metric = "ROC"
)

# ---- 6) Decision Tree (rpart) ----
set.seed(123)
model_tree <- train(
  stroke ~ ., data = train_data,
  method = "rpart",
  trControl = ctrl,
  preProcess = pp,
  tuneLength = 10,
  metric = "ROC"
)

# ---- 7) Random Forest ----
set.seed(123)
model_rf <- train(
  stroke ~ ., data = train_data,
  method = "rf",
  trControl = ctrl,
  preProcess = pp,
  tuneLength = 3,
  metric = "ROC"
)

# ---- 8) Compare via resamples (cross-validated ROC) ----
results <- resamples(list(Logistic = model_log,
                          DecisionTree = model_tree,
                          RandomForest = model_rf))
print(summary(results))
bwplot(results, metric = "ROC")

# ---- 9) Keep models for the next task (evaluation on holdout test set) ----
models <- list(Logistic = model_log,
               DecisionTree = model_tree,
               RandomForest = model_rf)

# =========================
# End: Task Two
# =========================

```




# Task Three: Evaluate and select prediction models

```{r}
# =========================
# Task Three: Evaluate & Select Prediction Models
# =========================

library(caret)
library(pROC)

set.seed(123)

# ---- 1) Predictions on test set ----
pred_log <- predict(model_log,  newdata = test_data)
pred_tree <- predict(model_tree, newdata = test_data)
pred_rf <- predict(model_rf, newdata = test_data)

# Probabilities (needed for ROC curves)
prob_log <- predict(model_log,  newdata = test_data, type = "prob")[, "Yes"]
prob_tree <- predict(model_tree, newdata = test_data, type = "prob")[, "Yes"]
prob_rf <- predict(model_rf, newdata = test_data, type = "prob")[, "Yes"]

# ---- 2) Confusion matrices ----
cm_log <- confusionMatrix(pred_log,  test_data$stroke, positive = "Yes")
cm_tree <- confusionMatrix(pred_tree, test_data$stroke, positive = "Yes")
cm_rf <- confusionMatrix(pred_rf,   test_data$stroke, positive = "Yes")

cat("\n--- Logistic Regression ---\n")
print(cm_log)
cat("\n--- Decision Tree ---\n")
print(cm_tree)
cat("\n--- Random Forest ---\n")
print(cm_rf)

# ---- 3) ROC curves + AUC ----
roc_log <- roc(test_data$stroke, prob_log, levels = c("No", "Yes"), direction = "<")
roc_tree <- roc(test_data$stroke, prob_tree, levels = c("No", "Yes"), direction = "<")
roc_rf <- roc(test_data$stroke, prob_rf, levels = c("No", "Yes"), direction = "<")

plot(roc_log, col = "blue", lwd = 2, main = "ROC Curves for Models")
plot(roc_tree, col = "purple", lwd = 2, add = TRUE)
plot(roc_rf, col = "darkorange", lwd = 2, add = TRUE)
legend("bottomright", legend = c(
  paste("Logistic (AUC =", round(auc(roc_log), 3), ")"),
  paste("Decision Tree (AUC =", round(auc(roc_tree), 3), ")"),
  paste("Random Forest (AUC =", round(auc(roc_rf), 3), ")")
), col = c("blue", "purple", "darkorange"), lwd = 2)

# ---- 4) Model comparison table ----
eval_results <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(cm_log$overall["Accuracy"], cm_tree$overall["Accuracy"], cm_rf$overall["Accuracy"]),
  Sensitivity = c(cm_log$byClass["Sensitivity"], cm_tree$byClass["Sensitivity"], cm_rf$byClass["Sensitivity"]),
  Specificity = c(cm_log$byClass["Specificity"], cm_tree$byClass["Specificity"], cm_rf$byClass["Specificity"]),
  F1 = c(cm_log$byClass["F1"], cm_tree$byClass["F1"], cm_rf$byClass["F1"]),
  AUC = c(auc(roc_log), auc(roc_tree), auc(roc_rf))
)

print(eval_results)

# ---- 5) Select best model ----
best_model_name <- eval_results$Model[which.max(eval_results$AUC)]
cat("\n✅ Best model based on AUC is:", best_model_name, "\n")

# =========================
# End: Task Three
# =========================

```


# Task Four: Deploy the prediction model

```{r}
# =========================
# Task Four: Deploy Prediction Model (fixed)
# =========================

library(caret)

# 1) Pick best by AUC (or swap to Sensitivity if you prefer)
best_model_name <- eval_results$Model[which.max(eval_results$AUC)]
cat("✅ Deploying best model by AUC:", best_model_name, "\n")

# 2) Map pretty names -> list keys in `models`
name_map <- c(
  "Logistic Regression" = "Logistic",
  "Decision Tree"       = "DecisionTree",
  "Random Forest"       = "RandomForest"
)

best_key <- unname(name_map[best_model_name])

if (is.na(best_key) || is.null(models[[best_key]])) {
  stop(sprintf(
    "Could not find model for '%s'. Available keys: %s",
    best_model_name, paste(names(models), collapse = ", ")
  ))
}

best_model <- models[[best_key]]

# 3) Save model
saveRDS(best_model, "best_stroke_model.rds")

# 4) Prediction helper (keeps factor levels, returns prob + label)
predict_stroke <- function(new_data, model = best_model, threshold = 0.5) {
  stopifnot(!is.null(model))
  new_data <- as.data.frame(new_data)

  # Ensure factor levels match training data where possible
  for (col in names(new_data)) {
    if (is.factor(new_data[[col]]) && col %in% names(train_data) && is.factor(train_data[[col]])) {
      new_data[[col]] <- factor(new_data[[col]], levels = levels(train_data[[col]]))
    }
  }

  probs <- predict(model, new_data, type = "prob")[, "Yes"]
  preds <- ifelse(probs >= threshold, "Stroke Risk", "No Stroke")
  data.frame(Prediction = preds, Probability = probs, row.names = NULL)
}

# 5) Example input (ensure values/levels exist in your data)
new_patient <- data.frame(
  gender          = factor("Male",            levels = levels(train_data$gender)),
  age             = 67,
  hypertension    = factor("1",               levels = levels(train_data$hypertension)),
  heart_disease   = factor("0",               levels = levels(train_data$heart_disease)),
  ever_married    = factor("Yes",             levels = levels(train_data$ever_married)),
  work_type       = factor("Private",         levels = levels(train_data$work_type)),
  Residence_type  = factor("Urban",           levels = levels(train_data$Residence_type)),
  avg_glucose_level = 228.69,
  bmi             = 36.6,
  smoking_status  = factor("formerly smoked", levels = levels(train_data$smoking_status))
)

# 6) Predict
predict_stroke(new_patient)


```




# Task Five: Findings and Conclusions


When I first looked at the dataset, one thing jumped out immediately: stroke cases are rare. Less than 5% of patients in the data actually had a stroke. That imbalance shaped the whole project, because in healthcare the real challenge is not just getting good accuracy — it’s making sure we don’t miss the people who are at risk.

A few patterns made sense right away. Age was the clearest signal — older patients had a much higher stroke risk. Hypertension and heart disease also stood out, along with average glucose level. Smoking status and BMI played a role too, but the relationships were a bit noisier.

I tested three different models:

Logistic Regression gave me a solid baseline. It’s simple and interpretable, which is a big plus when you want to explain risk factors to doctors or patients.

Decision Tree was the easiest to understand, but it didn’t perform very well overall.

Random Forest came out on top in terms of raw predictive power, with the best ROC-AUC and recall.

In the end, I saved the Random Forest as the “production” model because it did the best job of distinguishing stroke from non-stroke cases. But honestly, I think Logistic Regression deserves just as much credit — especially in healthcare, where interpretability matters almost as much as accuracy.

So what does this all mean?

It means stroke risk can be predicted with reasonable accuracy using fairly simple patient data. The model isn’t perfect (and it never will be), but it’s a helpful tool to flag high-risk patients early. If I were to take this project further, I’d want to add more clinical data, use techniques like SMOTE to balance the classes more effectively, and maybe build a Shiny app so clinicians could actually try it out.

For now, the main takeaway is this: machine learning can spot hidden stroke risks, but the real power comes from combining predictions with medical judgment.





























