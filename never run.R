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
if (!require("DT")) install.packages("DT")

library(shiny)
library(shinydashboard)
library(ggplot2)
library(plotly)
library(arules)
library(arulesViz)
library(cluster)
library(dplyr)
library(caret)
library(DT)

# Load the final dataset
final_data <- read.csv(file.choose())

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
          box(title = "Confusion Matrix Heatmap", width = 6, plotlyOutput("confusion_heatmap")),
          box(title = "Naive Bayes Accuracy", width = 6, verbatimTextOutput("nb_accuracy"))
        )
      ),
      # Regression Tab
      tabItem(
        tabName = "regression",
        fluidRow(
          box(title = "Actual vs Predicted Weekly Sales", width = 6, plotlyOutput("regression_plot")),
          box(title = "Residual Density Plot", width = 6, plotlyOutput("residual_density"))
        )
      ),
      # Clustering Tab
      tabItem(
        tabName = "clustering",
        fluidRow(
          box(title = "K-means Clustering", width = 6, plotOutput("kmeans_plot")),
          box(title = "Silhouette Plot", width = 6, plotOutput("silhouette_plot"))
        )
      ),
      # Association Rules Tab
      tabItem(
        tabName = "association_rules",
        fluidRow(
          box(title = "Rule Network", width = 6, plotOutput("rule_network")),
          box(title = "Rule Quality Scatterplot", width = 6, plotOutput("rule_quality_plot"))
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
  library(e1071)  # Load e1071 for naiveBayes
  # Split dataset into training and testing sets
  set.seed(123)
  sample_index <- sample(seq_len(nrow(final_data)), size = 0.70 * nrow(final_data))
  train_data <- final_data[sample_index, ]
  test_data <- final_data[-sample_index, ]
  
  # Naive Bayes Model
  nb_model <- naiveBayes(demand_level ~ Store + Dept + IsHoliday.x + Temperature + Fuel_Price, data = train_data)
  nb_predictions <- predict(nb_model, test_data)
  conf_matrix <- table(Predicted = nb_predictions, Actual = test_data$demand_level)
  
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
  
  # Naive Bayes Accuracy
  output$nb_accuracy <- renderPrint({
    accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
    paste("Naive Bayes Accuracy:", round(accuracy * 100, 2), "%")
  })
  
  # Linear Regression Model
  lm_model <- lm(Weekly_Sales ~ Store + Dept + IsHoliday.x + Temperature + Fuel_Price, data = train_data)
  test_data$Predictions <- predict(lm_model, test_data)
  
  # Regression Plot
  output$regression_plot <- renderPlotly({
    ggplot(test_data, aes(x = Weekly_Sales, y = Predictions)) +
      geom_point(color = "blue", alpha = 0.5) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
      labs(title = "Actual vs Predicted Weekly Sales", x = "Actual Sales", y = "Predicted Sales") +
      theme_minimal()
  })
  
  # Residual Density Plot
  output$residual_density <- renderPlotly({
    residuals <- test_data$Weekly_Sales - test_data$Predictions
    ggplot(data.frame(residuals), aes(x = residuals)) +
      geom_density(fill = "darkgreen", alpha = 0.7) +
      labs(title = "Density of Residuals", x = "Residuals", y = "Density") +
      theme_minimal()
  })
  
  # K-Means Clustering
  normalized_data <- scale(final_data[, c("Weekly_Sales", "Temperature", "Fuel_Price")])
  kmeans_model <- kmeans(normalized_data, centers = 3, nstart = 20)
  final_data$cluster <- kmeans_model$cluster
  
  output$kmeans_plot <- renderPlot({
    plot(final_data$Weekly_Sales, final_data$Temperature, col = final_data$cluster,
         main = "K-means Clustering", xlab = "Weekly Sales", ylab = "Temperature")
  })
  
  # Association Rules
  rules <- apriori(as(split(final_data$Dept, final_data$Store), "transactions"),
                   parameter = list(supp = 0.01, conf = 0.5))
  
  output$rule_network <- renderPlot({
    plot(rules, method = "graph", control = list(type = "items"))
  })
  
  output$rule_quality_plot <- renderPlot({
    plot(rules, measure = c("support", "confidence"), shading = "lift", method = "scatterplot")
  })
}

# Run the App
shinyApp(ui = ui, server = server)




















# Association Rules
rules <- apriori(as(split(final_data$Dept, final_data$Store), "transactions"),
                 parameter = list(supp = 0.01, conf = 0.5))

rule_network <- renderPlot({
  plot(rules, method = "graph", control = list(type = "items"))
})



