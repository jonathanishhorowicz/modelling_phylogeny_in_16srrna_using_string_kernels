library(ggplot2)
theme_set(theme_linedraw(base_size=12))
library(knitr)

cropped_ggsave <- function(save_path, plot=last_plot(), ...) {
  # save a plot then remove any surrounding whitespace
  cat('Saving plot to ', save_path, '\n')
  ggsave(save_path, plot=plot, ...)
  plot_crop(save_path)
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

equalise_x_and_y_limits <- function(plt) {
  # set x and y to same limits and aspect ratio to 1
  y_limits <- layer_scales(plt)$y$range$range
  x_limits <- layer_scales(plt)$x$range$range
  lower_lim <- c(min(x_limits[[1]], y_limits[[1]]))
  upper_lim <- c(max(x_limits[[2]], y_limits[[2]]))
  plt <- plt +
    xlim(c(lower_lim, upper_lim)) +
    ylim(c(lower_lim, upper_lim)) +
    coord_fixed()
  return(plt)
}