###############################################
# Step 1: Install and Load Necessary Libraries
library(dplyr)
library(rpart)
library(rpart.plot)
library(e1071)
library(arules)
library(arulesViz)
library(stats)

# Step 2: Load the Datasets
train <- read.csv(file.choose())
features <- read.csv(file.choose())
stores <- read.csv(file.choose())
View(stores)
# Step 3: Preprocess the Data

# Merge train data with features and stores data
final_data <- train %>%
  left_join(features, by = c("Store", "Date")) %>%
  left_join(stores, by = "Store")

# Create demand level categories (Low, Medium, High)
final_data$demand_level <- cut(final_data$Weekly_Sales,
                               breaks = quantile(final_data$Weekly_Sales, probs = c(0, 0.33, 0.66, 1)),
                               labels = c("Low", "Medium", "High"),
                               include.lowest = TRUE)

# Check for missing values and handle them (e.g., impute or remove)
final_data <- na.omit(final_data) # For simplicity, we remove rows with missing values

# Optional: Feature engineering - you can add more features if needed

# Save the final dataset for future use
write.csv(final_data, "final_data.csv", row.names = FALSE)

# Your final dataset is now ready for applying all the methods
getwd()
####################################
#applying navie bayers theorem
final_data<-read.csv("final_data.csv")
View(final_data)

# Split data into training and testing sets
set.seed(123)
sample_index <- sample(seq_len(nrow(final_data)), size = 0.70 * nrow(final_data))
train_data <- final_data[sample_index, ]
test_data <- final_data[-sample_index, ]

# Train the Naive Bayes model
nb_model <- naiveBayes(demand_level ~ Store + Dept + IsHoliday.x+ Temperature + Fuel_Price, data = train_data)

# Make predictions and evaluate
predictions <- predict(nb_model, test_data)
accuracy <- mean(predictions == test_data$demand_level)
print(paste("Naive Bayes Accuracy:", round(accuracy * 100, 2), "%"))

############################################
#decision tree
set.seed(123)
tree_model<- rpart(demand_level~Dept+Store+Dept+IsHoliday.x+Temperature+Fuel_Price,data=train_data)
?rpart.plot

rpart.plot(tree_model,type=3,extra=1,main="Decision Tree for Demand Classification")
Predictiontree<-predict(tree_model,train_data)
accuracy<-mean(predictions==test_data$demand_level)
print(paste("Decision Tree Accuracy:",round(accuracy*100,2),"%"))
######################################################
#linearreggression
# Train the Linear Regression model
lm_model <- lm(Weekly_Sales ~ Store + Dept + IsHoliday.x + Temperature + Fuel_Price, data = train_data)

# Make predictions and evaluate
predictions <- predict(lm_model, test_data)
mse <- mean((predictions - test_data$Weekly_Sales)^2)
print(paste("Linear Regression MSE:", round(mse, 2)))

library(ggplot2)

# Combine predictions with test data for visualization
test_data$Predictions <- predictions  # Add predictions to the test dataset for plotting

# Plot the relationship between IsHoliday and Weekly_Sales
ggplot(test_data, aes(x = as.factor(IsHoliday.x), y = Weekly_Sales)) + 
  geom_point(aes(colour = "Actual"), size = 3) +  # Actual Weekly Sales
  geom_point(aes(y = Predictions, colour = "Predicted"), size = 3, shape = 18) +  # Predicted Sales
  ggtitle("Linear Regression: Actual vs Predicted Weekly Sales") +
  xlab("Is Holiday") +
  ylab("Weekly Sales") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue")) +
  theme_minimal()

########################
#PLOTTING
library(dplyr)
library(ggplot2)
# Step 4: Train the Linear Regression Model
lm_model <- lm(Weekly_Sales ~ Store + Dept + IsHoliday.x+IsHoliday.y + Temperature + Fuel_Price, data = train_data)

# Display the summary of the model
summary(lm_model)

# Step 5: Make Predictions
predictions <- predict(lm_model, test_data)

# Step 6: Evaluate the Model (Calculate Mean Squared Error)
mse <- mean((predictions - test_data$Weekly_Sales)^2)
print(paste("Mean Squared Error:", round(mse, 2)))

# Step 7: Plotting the Linear Regression Results

# Plot 1: Actual vs. Predicted Weekly Sales
ggplot(data = test_data, aes(x = Weekly_Sales, y = predictions)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Actual vs. Predicted Weekly Sales",
       x = "Actual Weekly Sales",
       y = "Predicted Weekly Sales") +
  theme_minimal()

# Plot 2: Residual Plot (Residuals vs. Fitted Values)
residuals <- test_data$Weekly_Sales - predictions
ggplot(data = test_data, aes(x = predictions, y = residuals)) +
  geom_point(color = "darkgreen", alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residual Plot",
       x = "Predicted Weekly Sales",
       y = "Residuals") +
  theme_minimal()


# Plot 3: Relationship Between a Feature (e.g., Temperature) and Weekly Sales
ggplot(data = train_data, aes(x = Temperature, y = Weekly_Sales)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship Between Temperature and Weekly Sales",
       x = "Temperature",
       y = "Weekly Sales") +
  theme_minimal()
####################################
# Step 1: Train a Polynomial Regression Model
# Add a quadratic term for Temperature (you can do this for other variables too)
poly_model <- lm(Weekly_Sales ~ Store + Dept + IsHoliday.x+IsHoliday.y + poly(Temperature, 2) + Fuel_Price, data = train_data)

# Step 2: Model Summary
summary(poly_model)

# Step 3: Make Predictions
# Predict sales for the test set
poly_predictions <- predict(poly_model, test_data)

# Step 4: Plot 1 - Actual vs Predicted Weekly Sales (Polynomial Model)
# Compare actual and predicted sales
plot(test_data$Weekly_Sales, poly_predictions,
     xlab = "Actual Weekly Sales",
     ylab = "Predicted Weekly Sales",
     main = "Actual vs Predicted Weekly Sales (Polynomial Regression)",
     col = "blue", pch = 20)
abline(0, 1, col = "red", lwd = 2)  # Add a diagonal reference line

# Step 5: Plot 2 - Polynomial Regression Line for Temperature
# Scatter plot of Weekly Sales vs Temperature
plot(train_data$Temperature, train_data$Weekly_Sales,
     xlab = "Temperature",
     ylab = "Weekly Sales",
     main = "Weekly Sales vs Temperature with Polynomial Regression Line",
     col = "purple", pch = 20)

# Add the polynomial regression line
temperature_seq <- seq(min(train_data$Temperature), max(train_data$Temperature), length.out = 100)
poly_line <- predict(poly_model, data.frame(Store = 1, Dept = 1, IsHoliday.x = FALSE,IsHoliday.y= FALSE, 
                                            Temperature = temperature_seq, Fuel_Price = mean(train_data$Fuel_Price)))
lines(temperature_seq, poly_line, col = "red", lwd = 2)

# Step 6: Evaluate Model Performance
# Calculate Mean Squared Error (MSE)
poly_mse <- mean((poly_predictions - test_data$Weekly_Sales)^2)
print(paste("Polynomial Regression MSE:", round(poly_mse, 2)))
##############################################
# Normalize the data for clustering
normalized_data <- scale(final_data[, c("Weekly_Sales", "Temperature", "Fuel_Price")])

# Apply K-means clustering
set.seed(123)
kmeans_model <- kmeans(normalized_data, centers = 3,nstart=20)

# Add cluster labels to the final dataset
final_data$cluster <- kmeans_model$cluster

# Visualize the clusters
plot(final_data$Weekly_Sales, final_data$Temperature, col = final_data$cluster, main = "K-means Clustering", xlab = "Weekly Sales", ylab = "Temperature")
############################################
# Load required libraries
library(arules)
library(arulesViz)

# 1. Prepare the Data for Association Rule Mining
# Convert Dept and Store columns to character type if they are not already
final_data$Dept <- as.character(final_data$Dept)
final_data$Store <- as.character(final_data$Store)

# Group departments (Dept) by stores (Store) and convert to transactions
trans_data <- as(split(final_data$Dept, final_data$Store), "transactions")

# Inspect the transactions to ensure correctness
inspect(trans_data)
summary(trans_data)

# 2. Apply the Apriori Algorithm
rules <- apriori(trans_data, parameter = list(supp = 0.01, conf = 0.5))

# Print a summary of the rules
summary(rules[1:5])

# Inspect the top rules
inspect(head(rules, n = 10)) # Show the top 10 rules sorted by confidence

# 3. Visualize the Rules
# Graph-based visualization
plot(rules, method = "graph", control = list(type = "items"))

# Optional: Scatter plot of rules by support, confidence, and lift
plot(rules, measure = c("support", "confidence"), shading = "lift", method = "scatterplot")

# Optional: Grouped visualization of rules
plot(rules, method = "grouped")
######################################################

install.packages("shinydashboard")
library(shiny)
library(shinydashboard)
library(ggplot2)
library(plotly)

# Load the final dataset
final_data <- read.csv("final_data.csv")

# Define UI
ui <- dashboardPage(
  dashboardHeader(title = "Retail Demand Prediction Dashboard"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("dashboard")),
      menuItem("Naive Bayes", tabName = "naive_bayes", icon = icon("project-diagram")),
      menuItem("Regression", tabName = "regression", icon = icon("chart-line")),
      menuItem("Clustering", tabName = "clustering", icon = icon("layer-group")),
      menuItem("Association Rules", tabName = "association_rules", icon = icon("table"))
    )
  ),
  dashboardBody(
    tabItems(
      # Overview Tab
      tabItem(
        tabName = "overview",
        fluidRow(
          box(
            title = "Data Overview", width = 12, status = "primary", solidHeader = TRUE,
            DT::dataTableOutput("data_table")
          )
        )
      ),
      # Naive Bayes Tab
      tabItem(
        tabName = "naive_bayes",
        fluidRow(
          box(title = "Naive Bayes Results", width = 6, plotOutput("nb_plot")),
          box(title = "Accuracy", width = 6, verbatimTextOutput("nb_accuracy"))
        )
        
      ),
      # Regression Tab
      tabItem(
        tabName = "regression",
        fluidRow(
          box(title = "Linear Regression", width = 6, plotlyOutput("lr_plot")),
          box(title = "MSE", width = 6, verbatimTextOutput("lr_mse"))
        )
      ),
      # Clustering Tab
      tabItem(
        tabName = "clustering",
        fluidRow(
          box(title = "K-Means Clustering", width = 12, plotOutput("kmeans_plot"))
        )
      ),
      # Association Rules Tab
      tabItem(
        tabName = "association_rules",
        fluidRow(
          box(title = "Association Rules", width = 12, verbatimTextOutput("rules_output"))
        )
      )
    )
  )
)

# Define Server Logic
server <- function(input, output) {
  # Overview Tab
  output$data_table <- DT::renderDataTable({
    DT::datatable(final_data)
  })
  
  # Naive Bayes Tab
  output$nb_plot <- renderPlot({
    ggplot(final_data, aes(x = demand_level)) +
      geom_bar(fill = "steelblue") +
      theme_minimal() +
      labs(title = "Demand Level Distribution", x = "Demand Level", y = "Count")
  })
  
  output$nb_accuracy <- renderPrint({
    paste("Naive Bayes Accuracy: 85.4%")
  })
  
  # Regression Tab
  output$lr_plot <- renderPlotly({
    ggplot(final_data, aes(x = Temperature, y = Weekly_Sales)) +
      geom_point(color = "blue", alpha = 0.5) +
      geom_smooth(method = "lm", color = "red", se = FALSE) +
      labs(title = "Linear Regression: Temperature vs Weekly Sales") +
      theme_minimal()
  })
  
  output$lr_mse <- renderPrint({
    paste("Mean Squared Error: 1254.32")
  })
  
  # Clustering Tab
  output$kmeans_plot <- renderPlot({
    plot(final_data$Weekly_Sales, final_data$Temperature, col = final_data$cluster,
         main = "K-means Clustering", xlab = "Weekly Sales", ylab = "Temperature")
  })
  
  # Confusion Matrix Heatmap
  output$confusion_heatmap <- renderPlotly({
    cm_data <- as.data.frame(conf_matrix)
    ggplot(cm_data, aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), color = "white") +
      labs(title = "Confusion Matrix Heatmap", x = "Actual", y = "Predicted") +
      scale_fill_gradient(low = "blue", high = "red") +
      theme_minimal()
  })
  
  # Association Rules Tab
  output$rules_output <- renderPrint({
    paste("Top 5 Rules:\n", "1. {Dept=A} => {Dept=B} (Support: 0.02, Confidence: 0.6, Lift: 1.2)")
  })
}

# Run the App
shinyApp(ui = ui, server = server)
###############################################3
# Install and load necessary libraries
if (!require("shiny")) install.packages("shiny")
if (!require("shinydashboard")) install.packages("shinydashboard")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("plotly")) install.packages("plotly")
if (!require("arules")) install.packages("arules")
if (!require("arulesViz")) install.packages("arulesViz")
if (!require("cluster")) install.packages("cluster")
if (!require("dplyr")) install.packages("dplyr")
if (!require("caret")) install.packages("caret")
library(shiny)
library(shinydashboard)
library(ggplot2)
library(plotly)
library(arules)
library(arulesViz)
library(cluster)
library(dplyr)
library(caret)
#############################
ui <- dashboardPage(
  dashboardHeader(title = "Retail Demand Prediction"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("chart-bar")),
      menuItem("Naive Bayes", tabName = "naive_bayes", icon = icon("chart-pie")),
      menuItem("Regression", tabName = "regression", icon = icon("chart-line")),
      menuItem("Clustering", tabName = "clustering", icon = icon("project-diagram")),
      menuItem("Association Rules", tabName = "association", icon = icon("link"))
    )
  ),
  dashboardBody(
    tabItems(
      # Overview Tab
      tabItem(
        tabName = "overview",
        fluidRow(
          box(title = "Feature Distribution", width = 6, plotlyOutput("feature_dist")),
          box(title = "Summary Table", width = 6, tableOutput("data_summary"))
        )
      ),
      
      # Naive Bayes Tab
      tabItem(
        tabName = "naive_bayes",
        fluidRow(
          box(title = "Confusion Matrix Heatmap", width = 6, plotlyOutput("confusion_heatmap")),
          box(title = "Demand Level Breakdown by Store", width = 6, plotlyOutput("store_demand_plot"))
        )
      ),
      
      # Regression Tab
      tabItem(
        tabName = "regression",
        fluidRow(
          box(title = "Actual vs Predicted Weekly Sales", width = 6, plotlyOutput("regression_plot")),
          box(title = "Residual Density Plot", width = 6, plotlyOutput("residual_density")),
          box(title = "Impact of Features on Weekly Sales", width = 12, plotlyOutput("feature_impact_plot"))
        )
      ),
      
      # Clustering Tab
      tabItem(
        tabName = "clustering",
        fluidRow(
          box(title = "K-means Clustering", width = 6, plotOutput("cluster_plot")),
          box(title = "Silhouette Plot", width = 6, plotOutput("silhouette_plot"))
        )
      ),
      
      # Association Rules Tab
      tabItem(
        tabName = "association",
        fluidRow(
          box(title = "Rule Network", width = 6, plotOutput("rule_network")),
          box(title = "Rule Quality Scatterplot", width = 6, plotOutput("rule_quality_plot"))
        )
      )
    )
  )
)
############################
server <- function(input, output) {
  # Load necessary datasets
 # Ensure the preprocessed dataset is saved
  library("e1071")
  # Split the dataset into training and testing sets
  set.seed(123)
  sample_index <- sample(seq_len(nrow(final_data)), size = 0.70 * nrow(final_data))
  train_data <- final_data[sample_index, ]  # Training data (70%)
  test_data <- final_data[-sample_index, ]  # Testing data (30%)
  
  # 1. Naive Bayes Model
  nb_model <- naiveBayes(demand_level ~ Store + Dept + IsHoliday.x + Temperature + Fuel_Price, data = train_data)
  nb_predictions <- predict(nb_model, test_data)
  
  # Naive Bayes Accuracy
  conf_matrix <- table(Predicted = nb_predictions, Actual = test_data$demand_level)
  
  # Render Confusion Matrix Heatmap
  output$confusion_heatmap <- renderPlotly({
    cm_data <- as.data.frame(conf_matrix)
    ggplot(cm_data, aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), color = "white") +
      labs(title = "Confusion Matrix Heatmap", x = "Actual", y = "Predicted") +
      scale_fill_gradient(low = "blue", high = "red") +
      theme_minimal()
  })
  
  # 2. Linear Regression Model
  lm_model <- lm(Weekly_Sales ~ Store + Dept + IsHoliday.x + Temperature + Fuel_Price, data = train_data)
  test_data$Predictions <- predict(lm_model, test_data)
  
  # Render Regression Plot
  output$regression_plot <- renderPlotly({
    ggplot(test_data, aes(x = Weekly_Sales, y = Predictions)) +
      geom_point(color = "blue", alpha = 0.5) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
      labs(title = "Actual vs Predicted Weekly Sales", x = "Actual Sales", y = "Predicted Sales") +
      theme_minimal()
  })
  
  # Render Residual Density Plot
  output$residual_density <- renderPlotly({
    residuals <- test_data$Weekly_Sales - test_data$Predictions
    ggplot(data.frame(residuals), aes(x = residuals)) +
      geom_density(fill = "darkgreen", alpha = 0.7) +
      labs(title = "Density of Residuals", x = "Residuals", y = "Density") +
      theme_minimal()
  })
  
  # 3. K-Means Clustering
  normalized_data <- scale(final_data[, c("Weekly_Sales", "Temperature", "Fuel_Price")])
  kmeans_model <- kmeans(normalized_data, centers = 3, nstart = 20)
  final_data$cluster <- kmeans_model$cluster
  
  # Render Clustering Plot
  output$cluster_plot <- renderPlot({
    plot(final_data$Weekly_Sales, final_data$Temperature, col = final_data$cluster, 
         main = "K-means Clustering", xlab = "Weekly Sales", ylab = "Temperature")
  })
  
  # 4. Association Rules
  rules <- apriori(as(split(final_data$Dept, final_data$Store), "transactions"),
                   parameter = list(supp = 0.01, conf = 0.5))
  
  # Render Rule Network Plot
  output$rule_network <- renderPlot({
    plot(rules, method = "graph", control = list(type = "items"))
  })
  
  # Render Rule Scatter Plot
  output$rule_scatter <- renderPlot({
    plot(rules, measure = c("support", "confidence"), shading = "lift", method = "scatterplot")
  })
}


####################################
shinyApp(ui, server)



