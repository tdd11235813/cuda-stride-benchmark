library(ggplot2)
library(readr)
library(plyr)
library(dplyr)
library(scales)

find_results_dir <- function () {
    results_dir <- "results/"
    for(z in seq(3)) {
        if(dir.exists(results_dir))
            return(results_dir)
        results_dir <- paste0("../",results_dir)
    }
    stop("Could not find results directory.")
}

## Result directory with csv results
RESULT_PATH <- find_results_dir()

get_benchmark_flist <- function () {
    return(list.files(path=RESULT_PATH, pattern = ".csv$", recursive = TRUE))
}

open_benchmark_csv <- function (i,fnames,flabels){
    fname<-paste0(RESULT_PATH,fnames[i])
    #extracting measurements
    ## trim_ws does not trim \t, https://github.com/tidyverse/readr/issues/801
    local_frame <- read_delim(fname,skip=1,col_names=TRUE, trim_ws = TRUE, delim=",")
    local_frame$.file_id <- i
    colnames(local_frame) <- trimws(colnames(local_frame))
    colnames(local_frame) <- gsub(' ','_',colnames(local_frame), fixed=TRUE)
    colnames(local_frame) <- gsub("_(in_GB/s)",'',colnames(local_frame), fixed=TRUE)

    local_frame <- local_frame %>%
        mutate_all(funs(gsub("\t", "", .))) %>%
        mutate_at(vars(-matches("(dev_name|min_time|max_throughput)")),funs(as.numeric))
    
    ## truth <- sapply(local_frame,is.character)
    ## local_frame <- data.frame(cbind(sapply(local_frame[,truth],trimws,which="both"),local_frame[,!truth]))

    #local_frame$n <- trimws(local_frame$n)
    #local_frame$hardware <- trimws(local_frame$dev_name)

    if(nchar(flabels[i])>0)
        local_frame$hardware <- flabels[i]
    else
        local_frame$hardware <- paste0(local_frame$hardware," (Data ",i,")")

    lines <- readLines(fname)
    line_1 = lines[[1]]
    if(grepl("reduction-cub", line_1)) {
      vcol <- 'blocks_i/numSMs'
      local_frame <- local_frame %>% mutate(blocks_i = 2*numSMs)
      local_frame <- local_frame %>% mutate('blocks_i/numSMs' = 2)
      local_frame <- local_frame %>% mutate(blocks_i = n / 128) %>% mutate('blocks_i/numSMs' = n / numSMs / 128) %>% bind_rows(local_frame)
    }


    return(local_frame)
}

## this binds the data frames together
get_benchmark_data <- function(fnames,flabels) {
    result <- ldply(seq_along(fnames), .fun=open_benchmark_csv, fnames=fnames, flabels=flabels)
    return(result)
}

get_benchmark_header <- function(fname) {
    h <- read.csv(paste0(RESULT_PATH,fname), sep=",", header=F, nrows=1)
    return(h)
}

get_args_default <- function() {
    args <- list()
    args$n <- 0
    args$xmetric <- 'blocks_i'
    args$ymetric <- 'max_throughput'
    return(args)
}

get_benchmark_tables <- function(benchmark_data, args) {
    filter_n  <- 0

    if(nchar(args$n)>0 && args$n!='-') {
        filter_n <- as.integer(args$n)
    }

    if(grepl("^blocks_i$", args$xmetric))
       xlabel <- 'Number of Blocks'
    if(grepl("^blocks_i/numSMs$", args$xmetric))
       xlabel <- 'Number of Blocks per SM'
       
    if(grepl("time", args$ymetric))
        ylabel <- paste0(args$ymetric,"_[ms]")
    else if(grepl("throughput", args$ymetric))
        ylabel <- paste0(args$ymetric,"_[GB/s]")

    succeeded <- benchmark_data %>% filter(n == filter_n)

##############################################################################

    data_colnames = colnames(succeeded)

                                        # extracting ymetric expression
    ymetric_keywords = trimws(unlist(strsplit(args$ymetric,"[-|+|/|*|)|(]")))
    ymetric_expression = args$ymetric

                                        # creating expression
    for(i in 1:length(ymetric_keywords)) {

        indices = grep(ymetric_keywords[i],data_colnames)
        if( length(indices) > 0 && !is.null(ymetric_keywords[i]) && nchar(ymetric_keywords[i]) > 1){
            to_replace = paste("succeeded[,",indices[1],"]",sep="")
            cat(i,ymetric_keywords[i],"->",to_replace,"in",ymetric_expression,"\n")
            ymetric_expression = gsub(ymetric_keywords[i],to_replace,
                                      ymetric_expression)
        }
    }


                                        # creating metric of interest (moi)
    new_values = as.data.frame(eval(parse(text=ymetric_expression)))
    colnames(new_values) <- c("ymoi")

    name_of_ymetric = args$ymetric

    if( length(ymetric_keywords) == 1  ){
        name_of_ymetric = data_colnames[grep(ymetric_keywords[1], data_colnames)[1]]
    }

    if(!is.null(ylabel)) {

        if( nchar(ylabel) > 1){
            name_of_ymetric = gsub("_"," ",ylabel)
        }
    }
    cat("[ylabel] using ylabel: ",name_of_ymetric,"\n")

    succeeded_ymetric_of_interest  = new_values
################################################################################

##############################################################################
                                        # extracting xmetric expression
    if(any(grepl(paste0("^",args$xmetric),data_colnames)) == FALSE){

        stop(paste(args$xmetric, "for x not found in available columns \n",data_colnames,"\n"))
    }

    cat(paste(">>", dim(succeeded[args$xmetric]), "\n"), file=stderr())
    print(succeeded, file=stderr())
    succeeded_xmetric_of_interest <- succeeded[args$xmetric]
    name_of_xmetric <- args$xmetric
    if(!is.null(xlabel)) {

        if( nchar(xlabel) > 1){
            name_of_xmetric = xlabel
        }
    }
    colnames(succeeded_xmetric_of_interest) <- c("xmoi")
    succeeded_factors <- succeeded #  %>% select(-ends_with("]"))

    succeeded_reduced <- bind_cols(succeeded_factors,
                                   succeeded_xmetric_of_interest,
                                   succeeded_ymetric_of_interest)


    ## cols_to_consider <- Filter(function(i){ !(i %in% filtered_by || i == "id") },c(colnames(succeeded_factors),"xmoi"))
    ## cols_to_grp_by <- lapply(c(cols_to_consider,"id"), as.symbol)

    data_for_plotting <- succeeded_reduced
    ## %>%
    ##     group_by_(.dots = cols_to_grp_by) %>%
    ##     ##group_by(library, hardware, id, nx, ny, nz, xmoi) %>%
    ##     summarize( moi_mean = mean(ymoi),
    ##               moi_median = median(ymoi),
    ##               moi_stddev = sd(ymoi)
    ##               )
    data_for_plotting$ymoi <- as.numeric(gsub("[^0-9.]","",data_for_plotting$ymoi))
#### data2$y(x)/data1$y(x)
    tables <- list()
    tables$raw <- succeeded_reduced
    tables$reduced <- data_for_plotting
    tables$name_of_xmetric <- name_of_xmetric
    tables$name_of_ymetric <- name_of_ymetric

#    tables <- data_for_plotting[c('id','xmoi','moi_mean','moi_median','moi_stddev')]
    return(tables)
}

plot_benchmark <- function(tables,
                            aesthetics="TBlocksize,hardware",
                            usepoints=T,
                            noerrorbar=F,
                            nolegend=F,
                            usepointsraw=F,
                            freqpoly=F,
                            bins=200,
                            xlimit="",
                            ylimit="",
                            logx="-",
                            logy="-",
                            title="") {
    succeeded_reduced <- tables$raw
    data_for_plotting <- tables$reduced
    name_of_xmetric <- tables$name_of_xmetric
    name_of_ymetric <- tables$name_of_ymetric

    my_theme <-  theme_bw() + theme(axis.title.x = element_text(size=18),
                                    axis.title.y = element_text(size=18),
                                    axis.text.x = element_text(size=14),
                                    axis.text.y = element_text(size=14)#,
                                        #axis.text.x  = element_text()
                                   ,plot.margin = unit(c(8,10,1,1), "pt") # required otherwise labels are clipped in pdf output
                                    )
    my_theme <- my_theme + theme(#legend.title = element_blank(),
                                 legend.title = element_text(size=16, face="bold"),
                                 legend.text = element_text( size = 16),
                                 legend.position="bottom",
                                 legend.direction="vertical",
                                 legend.box ="horizontal",
                                 legend.box.just ="bottom",
                                 legend.background = element_rect(colour = 'white', fill = 'white', size = 0., linetype='dashed'),
                                 legend.key = element_rect(colour = 'white', fill = 'white', size = 0., linetype='dashed'),
                                 legend.key.width = unit(1.1, "cm"),
                                 panel.grid.major = element_line(color="#444444", size=0.5, linetype=3),
                                 panel.grid.minor = element_line(color="#555555", size=0.5, linetype=3)
                                 )

    aesthetics_from_cli <- strsplit(aesthetics,",")[[1]]

    aesthetics_keys   <- c("colour","shape","linetype")
    aesthetics_to_use <- aes(x=xmoi)
    aesthetics_length <- length(aesthetics_from_cli)
    n_items_per_aesthetics = c()
    counter = 1

    for(i in 1:length(aesthetics_keys)) {

        if( i <= aesthetics_length ){
            ## current_levels = eval(parse(text=paste("levels(as.factor(data_for_plotting$",
            ##                                 aesthetics_from_cli[i],"))",
            ##                                 sep="")))
            data_for_plotting[[ aesthetics_from_cli[i] ]] <- as.factor(data_for_plotting[[ aesthetics_from_cli[i] ]])
            succeeded_reduced[[ aesthetics_from_cli[i] ]] <- as.factor(succeeded_reduced[[ aesthetics_from_cli[i] ]])

            current_levels <- levels(data_for_plotting[[ aesthetics_from_cli[i] ]])

            n_items_per_aesthetics[counter] = length(current_levels)
            counter = counter + 1
            aesthetics_to_use[[aesthetics_keys[i]]] <- as.symbol(aesthetics_from_cli[i])
        }
    }

    if(freqpoly) {
        moi_plot <- ggplot(succeeded_reduced, aesthetics_to_use)
        moi_plot <- moi_plot + geom_freqpoly(bins=bins,size=1)
        name_of_ymetric <- "Frequency"
    } else if ( usepointsraw ) {
        ## cols_to_consider <- Filter(function(i){ !(i %in% filtered_by || i == "id" || i == "run") },c(colnames(succeeded_factors)))
        ## cols_to_grp_by <- lapply(c(cols_to_consider,"library"), as.symbol)
        ## cfs <- succeeded_reduced %>%
        ##     group_by_(.dots = cols_to_grp_by) %>%
        ##     summarize(moi_cf_a = t.test(ymoi)$conf.int[1],
        ##               moi_cf_b = t.test(ymoi)$conf.int[2]
        ##               )
        ## glimpse(cfs)
        ##    moi_plot <- ggplot(data_for_plotting, aesthetics_to_use)
        ##    moi_plot <- moi_plot + geom_point(aes(y=moi_mean),size=0.3,alpha=0.4)
        moi_plot <- ggplot(succeeded_reduced, aesthetics_to_use)
        ## moi_plot <- moi_plot + geom_hline(data=cfs,aes(yintercept=moi_cf_a,colour=library),alpha=1)
        ## moi_plot <- moi_plot + geom_hline(data=cfs,aes(yintercept=moi_cf_b,colour=library),alpha=1)
        moi_plot <- moi_plot + geom_point(aes(y=ymoi),size=1,alpha=1)
        ##    moi_plot <- moi_plot + geom_line(aes(y=moi_mean),size=.8)
        moi_plot <- moi_plot + scale_linetype_manual(values = c("solid","dotted","longdash","dashed")) + theme_bw()
    } else {
        print(aesthetics_to_use)
        moi_plot <- ggplot(data_for_plotting, ## aes(x=xmoi,
                           ##     #y=mean_elapsed_sec,
                           ##     color=library,
                           ##     linetype=hardware)
                           aesthetics_to_use
                           )
        moi_plot <- moi_plot + geom_line(aes(y=ymoi),size=1) #mean
        if( usepoints ) {
            moi_plot <- moi_plot + geom_point(aes(y=ymoi),size=3.5) #mean
        }
        ## if( noerrorbar == FALSE ) {
        ##     moi_plot <- moi_plot + geom_errorbar(aes(ymin = moi_mean - moi_stddev,
        ##                                              ymax = moi_mean + moi_stddev),
        ##                                          width=0.25, linetype =1)
        ## }
                                        #moi_plot <- moi_plot + scale_color_manual(name = "", values = c("red", 'blue'),labels=?)
        moi_plot <- moi_plot + scale_linetype_manual(values = c("solid","dotted","longdash","dashed")) #2,3,5,4,22,33,55,44))
    }

    ##

    moi_plot <- moi_plot + ylab(gsub("_"," ",name_of_ymetric)) + xlab(gsub("_"," ",name_of_xmetric))
    moi_plot <- moi_plot + my_theme

    if(nchar(title)>1)
        moi_plot <- moi_plot + ggtitle(title)

    str_to_numeric = function( string, sep ) {

        splitted = unlist(strsplit(string,sep))
        vec = sapply(splitted, function(x) as.numeric(x))
        return(vec)
    }

    ## ylimit_splitted = unlist(strsplit(opts[["ylimit"]],","))
    ## ylimit_pair = sapply(ylimit_splitted, function(x) as.numeric(x))
    ylimit_pair = str_to_numeric(ylimit, ",")
    xlimit_pair = str_to_numeric(xlimit, ",")

    if( length(ylimit_pair) == 2 ) {
        if(ylimit_pair[1] != 0 || ylimit_pair[2]!=0){
            cat("[ylimit] setting to ",paste(ylimit_pair),"\n")
            moi_plot <- moi_plot + ylim(ylimit_pair[1],ylimit_pair[2])
        }
    }

    if( length(xlimit_pair) == 2 ) {
        if(xlimit_pair[1] != 0 || xlimit_pair[2]!=0){
            cat("[xlimit] setting to ",paste(xlimit_pair),"\n")
            moi_plot <- moi_plot + xlim(xlimit_pair[1],xlimit_pair[2])
        }
    }


    if(nolegend){
        moi_plot <- moi_plot + theme(legend.position="none")
    }

    logx_value <- 1
    logy_value <- 1
    if(logx!="-")
        logx_value <- as.integer(logx)
    if(logy!="-")
        logy_value <- as.integer(logy)

    xmin <- min(data_for_plotting$xmoi)
    xmax <- max(data_for_plotting$xmoi)

    ymin <- min(data_for_plotting$ymoi) #mean
    ymax <- max(data_for_plotting$ymoi) #mean


    if(logy_value > 1) {

        breaks_y = function(x) logy_value^x
        format_expr_y = eval(parse(text=paste("math_format(",logy_value,"^.x)",sep="")))

        if(length(ylimit_pair) == 2 && (ylimit_pair[1] != 0 && ylimit_pair[2]!=0)){
            scale_structure = scale_y_continuous(
                limits = ylimit_pair,
                trans = log_trans(base=logy_value),
                breaks = trans_breaks(paste("log",logy_value,sep=""), breaks_y),
                labels = trans_format(paste("log",logy_value,sep=""), format_expr_y))
        } else {
            scale_structure = scale_y_continuous(
                trans = log_trans(base=logy_value),
                breaks = trans_breaks(paste("log",logy_value,sep=""), breaks_y),
                labels = trans_format(paste("log",logy_value,sep=""), format_expr_y))
            
        }

        moi_plot <- moi_plot + scale_structure
    }



    if(logx_value > 1) {

        breaks_x = function(x) logx_value^x
        format_expr_x = eval(parse(text=paste("math_format(",logx_value,"^.x)",sep="")))
        if(length(xlimit_pair) == 2 && (xlimit_pair[1] != 0 && xlimit_pair[2]!=0)){
            scale_x_structure = scale_x_continuous(
                limits = xlimit_pair,
                trans = log_trans(base=logx_value),
                breaks = trans_breaks(paste("log",logx_value,sep=""), breaks_x),
                labels = trans_format(paste("log",logx_value,sep=""), format_expr_x)
            )
        } else {
            scale_x_structure = scale_x_continuous(
                trans = log_trans(base=logx_value),
                breaks = trans_breaks(paste("log",logx_value,sep=""), breaks_x),
                labels = trans_format(paste("log",logx_value,sep=""), format_expr_x)
            )
            
        }

        moi_plot <- moi_plot + scale_x_structure
    }

    
    return(moi_plot)
}
