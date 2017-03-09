library(shiny)
library(plotly)
library(ggplot2)

shinyServer(
  function(input, output, session) {
    data(swiss)
    
    library(caret)
    model <- train(Fertility ~ ., data = swiss, method = 'glm')
    
    output$Agriculture <- renderPlotly({
      p1 <- ggplot(data=swiss, aes(x=Agriculture, y=Fertility)) +
        geom_point()+
        geom_smooth(method='lm')+
        geom_vline(xintercept = input$AgriSlider)
      p1 <- ptFunc(p1, reactive({input$numAgri})())
      ggplotly(p1)
    })
    output$Examination <- renderPlotly({
      p2 <- ggplot(data=swiss, aes(x=Examination, y=Fertility)) +
        geom_point()+
        geom_smooth(method='lm')+
        geom_vline(xintercept = input$examSlider)
      p2 <- ptFunc(p2, reactive({input$numExam})())
      ggplotly(p2)
    })
    output$Education <- renderPlotly({
      p3 <- ggplot(data=swiss, aes(x=Education, y=Fertility)) +
        geom_point()+
        geom_smooth(method='lm')+
        geom_vline(xintercept = input$eduSlider)
      p3 <- ptFunc(p3, reactive({input$numEdu})())
      ggplotly(p3)
    })
    output$Catholic <- renderPlotly({
      p4 <- ggplot(data=swiss, aes(x=Education, y=Catholic)) +
        geom_point()+
        geom_smooth(method='lm')+
        geom_vline(xintercept = input$cathSlider)
      p4 <- ptFunc(p4, reactive({input$numCath})())
      ggplotly(p4)
    })
    output$IM <- renderPlotly({
      p5 <- ggplot(data=swiss, aes(x=Infant.Mortality, y=Fertility)) +
        geom_point()+
        geom_smooth(method='lm')+
        geom_vline(xintercept = input$IMSlider)
      p5 <- ptFunc(p5, reactive({input$numIM})())
      ggplotly(p5)
    })
    
    prediction <- eventReactive(input$Apply, {
                    predict(model, data.frame(Agriculture = input$numAgri,
                                              Examination = input$numExam,
                                              Education = input$numEdu,
                                              Catholic = input$numCath,
                                              Infant.Mortality = input$numIM))
    })
    
    output$pred <- renderText({
      paste0(round(prediction(), 1), '%')
    })
    
    ptFunc <- function(plot, var){
      if (input$plotpoint %%2 == 1 & input$plotpoint != 0){
        data=data.frame(x=var,y=prediction())
        plot <- plot + geom_point(data=data, aes(x=x, y=y), color='red')
      }
      return(plot)
    }
    
  }
)





