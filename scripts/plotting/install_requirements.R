if(!('pacman' %in% installed.packages())) {
    cat('Installing pacman package from CRAN\n')
    install.packages("pacman", repos = "http://cran.us.r-project.org")
}

library(pacman)

# Install MASS and Matrix separately so we can specify the version - latest versions do not support R 4.1
if(!('MASS' %in% installed.packages())) {
    cat('Installing MASS package from CRAN\n')
    remotes::install_version('MASS', version = '7.3-57', repos = "http://cran.us.r-project.org")
}
if(!('Matrix' %in% installed.packages())) {
    cat('Installing Matrix package from CRAN\n')
    remotes::install_version('Matrix', version = '1.4-1', repos = "http://cran.us.r-project.org")
}

p_load(
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