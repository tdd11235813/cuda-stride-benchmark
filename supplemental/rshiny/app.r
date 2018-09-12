## requires packages DT, shiny, ...

library(ggplot2)
library(shiny)

source("helper.r", keep.source=TRUE)

flist1_selected <- ""
flist2_selected <- ""
tmpid <- 0

benchmark_flist <- get_benchmark_flist()

filter_by_tags <- function(flist, tags) {

    if( is.null(tags)==FALSE )
    {
        flist <- benchmark_flist
        matches <- Reduce(intersect, lapply(tags, grep, flist))
        flist <- flist[ matches ]
    }
    return(flist)
}


get_input_files <- function(input,datapath=T) {

    if(input$sData1=='User')
        files <- ifelse(datapath, input$file1$datapath, input$file1$name)
    else {
        files <- input$file1
        flist1_selected <<- input$file1
    }
    if(input$sData2=='User')
        files <- append(files, ifelse(datapath, input$file2$datapath, input$file2$name))
    else if(input$sData2=='original') {
        files <- append(files, input$file2)
        flist2_selected <<- input$file2
    }

    return(unique(unlist(files)))
}

get_args <- function(input) {

    args <- get_args_default()
    args$xmetric <- input$sXmetric
    args$ymetric <- input$sYmetric
    args$n <- input$sn
    return(args)

}

## Server

server <- function(input, output, session) {
    observe({
        input_files <- get_input_files(input)
        if (!is.null(input$sData1) && input$sData1!="none") {
            df_data <- get_benchmark_data(input_files,c(input$sCustomName1,input$sCustomName2))
            array_n <- df_data$n
        }
        if (!is.null(input$sData2) && input$sData2!="none") {
            df_data <- get_benchmark_data(input_files,c(input$sCustomName1,input$sCustomName2))
            array_n <- unique( c(array_n,df_data$n) )            
        }
        updateSelectInput(session, "sn", choices=array_n)
    })
    
    output$fInput1 <- renderUI({
        if (is.null(input$sData1))
            return()
        flist <- benchmark_flist
        flist <- filter_by_tags(flist, input$tags1) ## files matching tags like cuda p100 ...
        if(is.null(flist1_selected) || flist1_selected %in% flist == FALSE) ## if flist1_selected is not in (filtered) flist, disable it
            flist1_selected<<-""
        
        switch(input$sData1,
               "original" = selectInput("file1", "File", choices=flist, selected=flist1_selected),
               "User" = fileInput("file1", "File")
               )
    })

    output$fInput2 <- renderUI({
        if (is.null(input$sData2) || input$sData2=="none")
            return()
        flist <- benchmark_flist
        flist <- filter_by_tags(flist, input$tags2)
        if(is.null(flist2_selected)==TRUE || flist2_selected %in% flist == FALSE)
            flist2_selected<<-"" # <<- superassignment as we have a global variable here
        if(flist2_selected==flist1_selected && length(flist)>1)
            flist2_selected<<-flist[2]
        switch(input$sData2,
               "original" = selectInput("file2", "File", choices=flist, selected=flist2_selected),
               "User" = fileInput("file2", "File")
               )
    })

    output$sTable <- DT::renderDataTable(DT::datatable({

        if(is.null(input$file1))
            return()
        input_files <- get_input_files(input)
        args <- get_args(input)

        df_data <- get_benchmark_data(input_files,c(input$sCustomName1,input$sCustomName2))
        result <- get_benchmark_tables(df_data, args)

        return(result$reduced)
    }, style="bootstrap"))

    output$sTableRaw <- DT::renderDataTable(DT::datatable({

        if(is.null(input$file1))
            return()
        input_files <- get_input_files(input)

        df_data <- get_benchmark_data(input_files,c(input$sCustomName1,input$sCustomName2))

        return(df_data)
    }, style="bootstrap"))

    output$sPlot <- renderPlot({

        if(is.null(input$file1)) {
            return()
        }
        input_files <- get_input_files(input)
        args <- get_args(input)

        df_data <- get_benchmark_data(input_files,c(input$sCustomName1,input$sCustomName2))
        tables <- get_benchmark_tables(df_data, args)

        ## aesthetics
        aes <- c("TBlocksize","hardware","hardware")
        ## if(nlevels(as.factor(tables$reduced$hardware))>1)
        ##     aes <- append(aes,"hardware")
        ## if(input$sAes!="-")
        ##     aes <- append(aes,input$sAes)
        ## if(length(aes)<3)
        ##     aes <- append(aes,"library")
        aes_str <- paste(aes, collapse=",")

        freqpoly <- F
        usepointsraw <- F
        usepoints <- F
        noerrorbar <- T

        ## plot type
        if(input$sPlotType=="Histogram") {
            freqpoly <- T
            noerrorbar <- T
        } else if(input$sPlotType=="Points") {
            usepointsraw <- T
        } else {
            usepoints <- input$sUsepoints || length(aes)>2
            noerrorbar <- input$sNoerrorbar
        }
        ## if(input$sSpeedup==T) {
        ##     noerrorbar <- T
        ## }
        plot_benchmark(tables,
                        aesthetics = aes_str,
                        logx = input$sLogx,
                        logy = input$sLogy,
                        freqpoly = freqpoly,
                        bins = input$sHistBins,
                        usepoints = usepoints,
                        usepointsraw = usepointsraw,
                        noerrorbar = noerrorbar,
                        xlimit = input$sXlimit,
                        ylimit = input$sYlimit,
                        title = paste(args$n,"elements |",paste(input_files, collapse=" + "))
                        )
    })

    output$sPlotOptions <- renderUI({
        if(input$sPlotType == "Histogram")
            column(2, numericInput("sHistBins", "Bins", 200, min=10, max=1000))
        else if(input$sPlotType == "Lines") {
            fluidRow(column(1, checkboxInput("sUsepoints", "Draw Points", value=TRUE)),
                     column(2, checkboxInput("sNoerrorbar", "Disable Error-Bars")))
        }
    })

    output$sInfo <- renderUI({
        input_files <- get_input_files(input)
        header <- get_benchmark_header( input_files[1] )

        if(length(input_files)>1) {
            header2 <- get_benchmark_header( input_files[2] )
            
            wellPanel(
                br(),
                h4(input_files[1]),
                fluidRow(p(header)),
                h4(input_files[2]),
                fluidRow(p(header2))
            )
        } else {

            wellPanel(
                br(),
                h4(input_files[1]),
                fluidRow(p(header))
            )
        }
    })

    #
    output$sHint <- renderUI({
        if(input$sPlotType == "Histogram")
            p("Histograms help to analyze data of the validation code.", HTML("<ul><li>Use Time_* as xmetric for the x axis.</li><li>Probably better to disable log-scaling</li><li>If you do not see any curves then disable some filters.</li></ul>"))
        else if(input$sPlotType == "Lines")
            p("Lines are drawn by the averages including error bars.", HTML("<ul><li>If you see jumps then you should enable more filters or use the 'Inspect' option.</li><li>Points are always drawn when the degree of freedom in the diagram is greater than 2.</li><li>no error bars are shown when speedup option is enabled (speedup is computed on the averages)</li><li>when x-range or y-range is used '0' is only valid for non-logarithmic scales ('0,0' means automatic range)</li></ul>"))
        else if(input$sPlotType == "Points")
            p("This plot type allows to analyze the raw data by plotting each measure point. It helps analyzing the results of the validation code.")

    })
}






## User Interface

ui <- fluidPage(

    theme="simplex.min.css",
    tags$style(type="text/css",
               "label {font-size: 12px;}",
               "p {font-weight: bold;}",
               "h3 {margin-top: 0px;}",
               ".checkbox {vertical-align: top; margin-top: 0px; padding-top: 0px;}"
               ),

    h1("cuda-stride-benchmark | Benchmarks"),
    p("cuda-stride-benchmark is a benchmark to evaluate grid-stride and monolithic kernel performance. Get ",
      a(href="https://github.com/tdd11235813/cuda-stride-benchmark", "cuda-stride-benchmark on github.")),
    hr(),

    wellPanel(
        h3("Data"),
        p("Data is provided here."),
        fluidRow(
            column(6, wellPanel( fluidRow(
                          column(3, selectInput("sData1", "Data 1", c("original", "User"))),
                          column(9, uiOutput("fInput1"))
                      ),
                      fluidRow(
                          checkboxGroupInput("tags1", "Tags",
                                             c(
					       "k80"="K80",
					       "p100"="P100",
					       "v100"="V100"
                                               ),
                                             inline=T
                                             )),
                      fluidRow(
                          column(8,textInput("sCustomName1","Custom Library Name (leave it empty for default label)",""))
                      )
                      )),
            column(6, wellPanel( fluidRow(
                          column(3, selectInput("sData2", "Data 2", c("original", "User", "none"), selected="original")),
                          column(9, uiOutput("fInput2"))
                      ),
                      fluidRow(
                          checkboxGroupInput("tags2", "Tags",
                                             c(
                                               "k80"="K80",
					       "p100"="P100",
					       "v100"="V100"
                                               ),
                                             inline=T
                                             )),
                      fluidRow(
                          column(8,textInput("sCustomName2","Custom Library Name (leave it empty for default label)",""))
                      )
                      ))
        ),

        h3("Filtered by"),
        wellPanel(
        fluidRow(
#                 uiOutput("sn")
            column(4, selectInput("sn", "n", c(""), selected="")),
            column(4, selectInput("sXmetric", "xmetric", c("blocks_i","blocks_i/numSMs"))),
            column(4, selectInput("sYmetric", "ymetric", c("max_throughput","min_time")))
            )
        )
    ),

    tabsetPanel(
        ## Plot panel
        tabPanel("Plot",

                 br(),
                 plotOutput("sPlot",height=500),
                 br(),
                 wellPanel(
                     h3("Plot Options"),
                     fluidRow(
                         column(3, selectInput("sPlotType", "Plot type", c("Lines","Histogram"), selected="Lines")),
                         column(1, selectInput("sLogx", "Log-X", c("-","2","10"), selected="2")),
                         column(1, selectInput("sLogy", "Log-Y", c("-","2","10"), selected="-")),
                         column(1, textInput("sXlimit", "x-range", "0,0")),
                         column(1, textInput("sYlimit", "y-range", "0,0")),
                         column(1, checkboxInput("sNotitle", "Disable Title")),
                         uiOutput("sPlotOptions")
                     ),
                     uiOutput("sHint"))),
        ## Table panel
        tabPanel("Table",
                 
                 br(),
                 DT::dataTableOutput("sTable"),
                 p("A table aggregates the data and shows the minimum of the runs for each benchmark."),
                 div(HTML("<ul><li>xmoi: xmetric of interest (xmetric='nbytes' -> signal size in MiB)</li><li>ymoi: ymetric of interest</li></ul>"))
                 ),
        ## Table panel
        tabPanel("Raw Data",
                 
                 br(),
                 DT::dataTableOutput("sTableRaw")
                 ),
        tabPanel("Info",
                 
                 br(),
                 uiOutput("sInfo")                 
                 )
    ),
    hr(),
    
    ## fluidRow(verbatimTextOutput("log"))
    ##    mainPanel(plotOutput("distPlot"))
    ##  )
    
    span("This tool is powered by R Shiny Server.")
)

## will look for ui.R and server.R when reloading browser page, so you have to run
## R -e "shiny::runApp('~/shinyapp')"
shinyApp(ui = ui, server = server)

