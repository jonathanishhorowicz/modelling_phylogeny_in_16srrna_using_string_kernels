if(!('pacman' %in% installed.packages())) {
    cat('Installing pacman package from CRAN\n')
    install.packages("pacman", repos = "http://cran.us.r-project.org")
}
    
library(pacman)
pacman::p_load(
    ggplot2,
    tidyr,
    dplyr,
    data.table,
    tidyr,
    stringr,
    ggpubr,
    ggforce,
    ggh4x,
    lemon,
    cowplot,
    forcats,
    binom,
    tibble
)