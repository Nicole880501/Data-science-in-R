# 安裝及匯入套件
install.packages("dplyr")
install.packages("ggplot2")
install.packages("caret")
install.packages("rpart")
install.packages("reshape2")
install.packages("pROC")
install.packages("e1071")
install.packages("caret")
install.packages("devtools")
install.packages("randomForest")
install.packages("epiDisplay")
devtools::install_github("selva86/InformationValue")

library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(reshape2)
library(pROC)
library(e1071)
library(InformationValue)
library(class)
library(rpart.plot)
library(scales)
library(randomForest)
library(epiDisplay)


# 讀取資料 ----
path <- "C:/Nicole/KMU/112-2/資料科學與統計特論/Project/diabetes_binary.csv"
data <- read.csv(path, sep=",")

# 保留原始資料
original_data <- data

# 設置隨機種子
set.seed(123)

# 計算描述性統計 ----
summary(data)

# 將連續性資料抓出來
continuous_col <- c("BMI", "MentHlth", "PhysHlth")
# 類別型資料
data <- data[, !colnames(data) %in% continuous_col]

# 資料視覺化 ----
plots <- list()

# 針對所有資料對target做分組畫圖
# 連續型
for (column in continuous_col) {
  p <- ggplot(original_data, aes_string(x = column, fill = "Diabetes_binary")) +
    geom_density(alpha = 0.5) +
    labs(title = paste("Density Plot of", column, "by Diabetes Status"), x = column, y = "Density", fill = "Diabetes Status")
  plots[[length(plots) + 1]] <- p
}

# 類別型
for (column in setdiff(names(data), "Diabetes_binary")) {
  p <- ggplot(data, aes_string(x = "Diabetes_binary", fill = paste("factor(", column, ")", sep=""))) +
    geom_bar(position = "dodge") +
    labs(title = paste("Count of Diabetes Status by", column), x = "Diabetes Status", y = "Count", fill = column) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate X labels if needed
  plots[[length(plots) + 1]] <- p
}

# 輸出圖
for (p in plots) {
  print(p)
}


# 熱力圖 ----
# 重新匯入資料
data <- read.csv(path, sep=",")
# 計算數值欄位的相關矩陣
numeric_data <- data[, sapply(data, is.numeric)]
cor_matrix <- cor(numeric_data, use = "complete.obs")

# 將相關矩陣轉換為長格式
melted_cor_matrix <- melt(cor_matrix)

# 繪製熱力圖並顯示相關係數數值
p <- ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#2c7bb6", high = "#d7191c", mid = "#ffffbf", midpoint = 0, limit = c(-1, 1), space = "Lab", name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) +
  coord_fixed() +
  labs(title = "Heatmap of Correlation Matrix", x = "", y = "") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3)

print(p)


# 將原始資料欄位轉類別型資料 ----
data <- original_data
# 使用迴圈將特定欄位轉換為factor
for (col in names(data)) {
  if (!(col %in% continuous_col)) {
    data[[col]] <- as.factor(data[[col]])
  }
}

# 將Diabetes_binary轉換為factor並確保因子層次名稱是有效的R變數名稱
data$Diabetes_binary <- factor(data$Diabetes_binary, levels = c(0, 1), labels = c("No", "Yes"))


# 卡方檢定 ----
factor_cols <- setdiff(names(data), continuous_col)
factor_cols <- setdiff(factor_cols, "Diabetes_binary")

cat("Chi-Square Test Results:\n")
for (col in factor_cols) {
  test_result <- chisq.test(table(data[[col]], data$Diabetes_binary))
  cat(paste("Column:", col, "\n"))
  print(test_result)
  cat("\n")
}

# T檢定 ----
cat("T-Test Results:\n")
for (col in continuous_col) {
  test_result <- t.test(data[[col]] ~ data$Diabetes_binary)
  cat(paste("Column:", col, "\n"))
  print(test_result)
  cat("\n")
}


# 切分數據集為訓練集和測試集 ----
# 設置隨機種子
set.seed(123)
train_indices <- createDataPartition(data$Diabetes_binary, p = 0.7, list = FALSE)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]


# 處理類別不平衡，隨機上下採樣 ----
# 分離兩類數據
data_0 <- train_data %>% filter(Diabetes_binary == 'No')
data_1 <- train_data %>% filter(Diabetes_binary == 'Yes')

# 計算平衡後的樣本數（兩類數據數量的平均值）
target_size <- round((nrow(data_0) + nrow(data_1)) / 2)

# 下採樣和上採樣
data_0_balanced <- data_0 %>% sample_n(target_size, replace = nrow(data_0) < target_size)
data_1_balanced <- data_1 %>% sample_n(target_size, replace = nrow(data_1) < target_size)

# 合併平衡後的數據
balanced_data <- bind_rows(data_0_balanced, data_1_balanced)

# 檢查類別平衡
table(balanced_data$Diabetes_binary)

# 顯示平衡後的柱狀圖
p <- ggplot(balanced_data, aes(x = factor(Diabetes_binary), fill = factor(Diabetes_binary))) +
  geom_bar() +
  labs(title = "Bar Plot of Diabetes_binary After Balancing", x = "Diabetes_binary", y = "Count") +
  scale_fill_discrete(name = "Diabetes_binary") +
  scale_y_continuous(labels = comma) +  # 使用 scales::comma 來格式化 Y 軸標籤
  theme_minimal()

print(p)



# 模型一：訓練決策樹模型 ----
# 設置隨機種子
set.seed(123)
dtree <- rpart(Diabetes_binary ~ ., data = balanced_data, method = "class")
summary(dtree)

# 顯示分類圖
prp(dtree,         # 模型
    # number of correct classifications / number of observations in that node
    extra=2)

## 決策樹模型特徵重要性 ----
importance <- dtree$variable.importance
print(importance)

# 從dtree模型獲取特徵重要性數據
importance <- dtree$variable.importance
importance_df <- as.data.frame(importance)  # 將特徵重要性轉換為數據框架
importance_df$Feature <- rownames(importance_df)  # 從行名創建特徵名欄位
colnames(importance_df)[1] <- "Importance"  # 重新命名重要性欄位

# 將重要性數值標準化成機率值
importance_df$Importance <- importance_df$Importance / sum(importance_df$Importance)  # 標準化處理

# 按重要性機率值排序特徵
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE),]

# 繪製特徵重要性圖
p <- ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance, fill = Feature)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Importance, 4)), vjust = -0.3, size = 3.5) +  # 添加數值標籤
  labs(title = "Feature Importance for Diabetes_binary", x = "Features", y = "Importance (Probability)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # X 軸標籤傾斜以改善可讀性

print(p)

## 決策樹模型評估 ----
# 預測類別
test_data$dtree_pred<-predict(dtree, test_data, type="class")
# 預測機率值畫ROC
test_data$dtree_value<-predict(dtree, test_data,  type="prob")[,2]

# 決策樹找最佳切點
(optCutOff_dtree <- optimalCutoff(test_data$Diabetes_binary, test_data$dtree_value)[1])
test_data$dtree_pred_opt<-ifelse(test_data$dtree_value>=optCutOff_dtree,1,0)

# 看預測及實際的差異in test_data
(confus_matrix_dtree<-table(x=test_data$Diabetes_binary,y=test_data$dtree_pred))
(confus_matrix_dtree_opt<-table(x=test_data$Diabetes_binary,y=test_data$dtree_pred_opt))

# 決策樹 ROC
rf.roc<-roc(test_data$Diabetes_binary,test_data$dtree_value, plot=TRUE, print.auc = TRUE, legacy.axes=TRUE)

# 決策樹 accuracy
(accuracy_dtree <- sum(diag(confus_matrix_dtree)) / sum(confus_matrix_dtree))
(accuracy_dtree_opt <- sum(diag(confus_matrix_dtree_opt)) / sum(confus_matrix_dtree_opt))

(precision_dtree <- confus_matrix_dtree[2, 2] / sum(confus_matrix_dtree[, 2]))
(recall_dtree <- confus_matrix_dtree[2, 2] / sum(confus_matrix_dtree[2, ]))
(f1_score_dtree <- 2 * precision_dtree * recall_dtree / (precision_dtree + recall_dtree))



# 模型二：訓練隨機森林模型 ----
# 設置隨機種子
set.seed(123)
rf_model <- randomForest(Diabetes_binary ~ ., data = balanced_data)
summary(rf_model)

# 畫error圖
plot(rf_model)

## 隨機森林特徵重要性 ----
# 那個變數重要
importance(rf_model)

## 隨機森林模型評估 ----
# 預測類別
test_data$rf_pred<-predict(rf_model, test_data)

# 預測機率值畫ROC
test_set_rf_prob<-predict(rf_model, test_data,  type="prob")
test_data$rf_value<-test_set_rf_prob[,2]


# 隨機森林ROC
rf.roc<-roc(test_data$Diabetes_binary,test_data$rf_value, plot=TRUE, print.auc = TRUE, legacy.axes=TRUE)


# 隨機森林找最佳切點
(optCutOff_rf <- optimalCutoff(test_data$Diabetes_binary, test_data$rf_value)[1])
test_data$rf_pred_opt<-ifelse(test_data$rf_value>=optCutOff_rf,1,0)


# 看預測及實際的差異in test_data
(confus_matrix_rf<-table(real=test_data$Diabetes_binary,predicted=test_data$rf_pred))
(confus_matrix_rf_opt<-table(real=test_data$Diabetes_binary,predicted=test_data$rf_pred_opt))

# 搭配plot看
plot(rf_model)

# 隨機森林 accuracy
(accuracy_rf <- sum(diag(confus_matrix_rf)) / sum(confus_matrix_rf))
(accuracy_rf_opt <- sum(diag(confus_matrix_rf_opt)) / sum(confus_matrix_rf_opt))

(precision_rf <- confus_matrix_rf[2, 2] / sum(confus_matrix_rf[, 2]))
(recall_rf <- confus_matrix_rf[2, 2] / sum(confus_matrix_rf[2, ]))
(f1_score_rf <- 2 * precision_rf * recall_rf / (precision_rf + recall_rf))



# 模型三：訓練Logistic模型 ----
# 設置隨機種子
set.seed(123)
log_model <- glm(Diabetes_binary ~ ., data = balanced_data, family = binomial(link = "logit"))
summary(log_model)

# 使用 logistic.display 函數顯示模型結果
logistic.display(log_model)

## Logistic模型特徵重要性----
# 計算係數的指數，這對應於每個特徵的機率比
exp_coefs <- exp(coef(log_model))
importance_df <- as.data.frame(exp_coefs, row.names = NULL)
importance_df$Feature <- rownames(importance_df)
colnames(importance_df)[1] <- "Odds Ratio"

# 按Odds Ratio的絕對值降序排序特徵
importance_df <- importance_df[order(abs(importance_df$`Odds Ratio`), decreasing = TRUE), ]

# 繪製特徵重要性圖
p <- ggplot(importance_df, aes(x = reorder(Feature, -`Odds Ratio`), y = `Odds Ratio`)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = sprintf("%.2f", `Odds Ratio`)), vjust = -0.3, size = 3.5) +
  labs(title = "Feature Importance in Logistic Regression Model", x = "Features", y = "Odds Ratio") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9)) 

print(p)

## Logistic模型評估 ----
test_data$logit_value<-predict(log_model, test_data, type="response")

#找logistic regrssion roc最佳切點
optCutOff <- optimalCutoff(test_data$Diabetes_binary, test_data$logit_value)[1] 
test_data$logit_pred<-ifelse(test_data$logit_value>=optCutOff,1,0)


# Logistic ROC
logit.roc<-roc(test_data$Diabetes_binary, test_data$logit_value)
plot(logit.roc, print.auc = TRUE)

# 看預測及實際的差異in test_data
(confus_matrix_logit<-table(x=test_data$Diabetes_binary,y=test_data$logit_pred))

# 算accuracy及其他評估指標
(accuracy_logit <- sum(diag(confus_matrix_logit)) / sum(confus_matrix_logit))
(precision_logit <- confus_matrix_logit[2, 2] / sum(confus_matrix_logit[, 2]))
(recall_logit <- confus_matrix_logit[2, 2] / sum(confus_matrix_logit[2, ]))
(f1_score_logit <- 2 * precision_logit * recall_logit / (precision_logit + recall_logit))



## 加入Logistic模型自定義閾值 ----
# 設定自定義閾值
threshold <- 0.6

# 使用自定義閾值進行預測
test_data$logit_pred_custom <- ifelse(test_data$logit_value >= threshold, 1, 0)

# 看預測及實際的差異 in test_data（使用自定義閾值）
(confus_matrix_logit_custom <- table(x = test_data$Diabetes_binary, y = test_data$logit_pred_custom))

# 算accuracy及其他評估指標（使用自定義閾值）
(accuracy_logit_custom <- sum(diag(confus_matrix_logit_custom)) / sum(confus_matrix_logit_custom))
(precision_logit_custom <- confus_matrix_logit_custom[2, 2] / sum(confus_matrix_logit_custom[, 2]))
(recall_logit_custom <- confus_matrix_logit_custom[2, 2] / sum(confus_matrix_logit_custom[2, ]))
(f1_score_logit_custom <- 2 * precision_logit_custom * recall_logit_custom / (precision_logit_custom + recall_logit_custom))

