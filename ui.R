library(shiny)
library(plotly)
library(ggplot2)
library(shinythemes)

shinyUI(
  navbarPage(
    title = 'JM Analytics', 
    theme = shinytheme('darkly'), 
    tabPanel(
      p(icon('area-chart'), 'Explore the data'), 
      sidebarPanel(
        width=8, 
###############################################        
        fluidRow(
          column(12, align = 'center', plotlyOutput('Agriculture', width = '50%', height = '180px'))
        ), 
##############
        fluidRow(
          column(12, align = 'center', sliderInput('AgriSlider', '% Male invovled in agriculture:', 0, 100, 50, width = '50%'))
        ), 
###############################################  
        fluidRow(
          column(12, 
                 splitLayout(cellwidths = c('50%', '50%'), 
                                 plotlyOutput('Examination', height = '180px'), 
                                 plotlyOutput('Education', height = '180px'))
          )
        ),
##############
        fluidRow(
          column(12, align='center', 
                 splitLayout(cellwidths = c('50%', '50%'), 
                             sliderInput('examSlider', '% drafees with best score:', 0, 100, 50), 
                             sliderInput('eduSlider', '% draftees beyond primary school', 0, 100, 50))
                 )
        ),
###############################################
        fluidRow(
          column(12, 
                 splitLayout(cellwidths = c('50%', '50%'), 
                             plotlyOutput('Catholic', height = '180px'), 
                             plotlyOutput('IM', height = '180px'))
          )
        ),
##############
        fluidRow(
          column(12, align='center', 
                 splitLayout(cellwidths = c('50%', '50%'), 
                             sliderInput('cathSlider', '% Catholic:', 0, 100, 50), 
                             sliderInput('IMSlider', '% births live less than 1 year:', 0, 100, 50))
          )
        )
###############################################
      ), 
    mainPanel(width=4, 
      tabsetPanel(
        tabPanel(p(icon('question-circle'), 'Introduction'), 
                 h2('About the data:'), 
                 h3('Standardized fertility measure and socio-economic indicators for each of 47 French-speaking provinces of Switzerland at about 1888'),
                 h5('Fertility (Ig):        Common standardised fertility measure'), 
                 h5('Agriculture (%):       % of males involved in agriculture as occupation'), 
                 h5('Examination (%):       % draftees receiving highest mark on army examination'), 
                 h5('Education (%):         % education beyond primary school for draftees'), 
                 h5('Catholic (%):          % catholic (as opposed to protestant'), 
                 h5('Infant Mortality (%):  Live births who live less than 1 year')
                 ), 
        tabPanel(p(icon('gear'), 'GLM'),
                 h4('Configure settings to see GLM modelling results:'), 
                 numericInput('numAgri', 'Agriculture', value = 50, min = 0, max = 100, step = 5), 
                 numericInput('numExam', 'Examination', value = 50, min = 0, max = 100, step = 5), 
                 numericInput('numEdu', 'Education', value = 50, min = 0, max = 100, step = 5), 
                 numericInput('numCath', 'Catholic', value = 50, min = 0, max = 100, step = 5), 
                 numericInput('numIM', 'Infant Mortality', value = 50, min = 0, max = 100, step = 5), 
                 actionButton('Apply', 'Apply'),
                 actionButton('plotpoint', 'Show/Hide point on charts'), 
                 h3('Predicted Fertility:'), 
                 verbatimTextOutput("pred")
                 )
      )
    )
    )
  )
)