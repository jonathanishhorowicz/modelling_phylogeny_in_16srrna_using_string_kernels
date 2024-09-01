library(ggplot2)
theme_set(theme_linedraw(base_size=12))
library(RColorBrewer)

####################################################################################
# utility functions for plots
####################################################################################


cropped_ggsave <- function(save_path, plot=last_plot(), ...) {
  ggsave(save_path, plot=plot, ...)
  knitr::plot_crop(save_path)
}

get_plot_theme <- function() {
  theme(
    strip.background=element_blank(),
    strip.text=element_text(size=24, colour="black"),
    panel.grid.major=element_line(linewidth=0.01, colour="grey50"),
    panel.grid.minor=element_line(linewidth=0.01, colour="grey50"),
    legend.title=element_text(size=24),
    legend.text=element_text(size=20),
    axis.text=element_text(size=20),
    axis.title=element_text(size=24),
    panel.spacing.x=unit(-10, "pt"),
    panel.spacing.y=unit(-5, "pt")
  )
}

####################################################################################
####################################################################################