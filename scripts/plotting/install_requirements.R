# This script is designed to run in R 4.1.3

if(!('pacman' %in% installed.packages())) {
    cat('Installing pacman package from CRAN\n')
    install.packages("pacman", repos = "http://cran.us.r-project.org")
}

library(pacman)

# Install MASS, Matrix. MatrixModels and pbkrtest separately so we can specify the version -
# their latest versions do not support R 4.1
if(!('MASS' %in% installed.packages())) {
    cat('Installing MASS package from CRAN\n')
    remotes::install_version('MASS', version = '7.3-57', repos = "http://cran.us.r-project.org")
}
if(!('Matrix' %in% installed.packages())) {
    cat('Installing Matrix package from CRAN\n')
    remotes::install_version('Matrix', version = '1.4-1', repos = "http://cran.us.r-project.org")
}
if(!('pbkrtest' %in% installed.packages())) {
    cat('Installing pbkrtest package from CRAN\n')
    remotes::install_version('pbkrtest', version = '0.5.1', repos = "http://cran.us.r-project.org")
}

if(!('MatrixModels' %in% installed.packages())) {
    cat('Installing MatrixModels package from CRAN\n')
    remotes::install_version('MatrixModels', version = '0.5-0', repos = "http://cran.us.r-project.org")
}

p_load(
    tidyr,
    dplyr,
    data.table,
    tidyr,
    stringr,
    forcats,
    binom,
    tibble,
    ggpubr,
    ggh4x,
    ggforce,
    lemon,
    cowplot
)