library(ggplot2)
theme_set(theme_linedraw(base_size=12))
# library(ggtext)
library(RColorBrewer)

####################################################################################
# utility functions for plots
####################################################################################

feather_save <- function(x, savepath, ...) {
  x %>%
    as.data.frame() %>%
    rownames_to_column("rownames") %>%
    arrow::write_feather(savepath, ...)
}

feather_load <- function(savepath, ...) {
  arrow::read_feather(savepath, ...) %>%
    column_to_rownames("rownames")
}

cropped_ggsave <- function(save_path, plot=last_plot(), ...) {
  ggsave(save_path, plot=plot, ...)
  knitr::plot_crop(save_path)
}

thesis_theme <- function() {
  theme(strip.background=element_blank(),
        strip.text=element_text(colour="black"),
        text=element_text(size=16),
        panel.grid.major.y=element_line(size=0.01),
        panel.grid.major.x=element_blank(),
        legend.text=element_markdown())
}

get_prior_type <- function(x) {
  key <- str_extract(x, "rbf|matern|tree")
  prior_types <- c(
    matern="No phylogeny",
    rbf="No phylogeny",
    tree="Phylogeny"
  )
  return(prior_types[ key ])
}

get_kernel_type <- function(x) {
  key <- str_extract(x, "rbf|matern|string|unifrac")
  prior_types <- c(
    matern="no_phylogeny",
    rbf="no_phylogeny",
    string="phylogeny",
    unifrac="phylogeny"
  )
  return(prior_types[ key ])
}

prettify <- function(df) {
  df %>%
    mutate(kernel=recode(kernel,
                         "unweighted-unifrac-centre"="UniFrac (UW)",
                         "weighted-unifrac-centre"="UniFrac (W)",
                         rbf="RBF",
                         matern32="Matern32",
                         string="String"),
           SIGMA_SQ=paste0("Noise variance=", SIGMA_SQ),
           dataset=str_replace(dataset, "__", "\n"),
           prior_type=paste0(prior_type, " in generative model"),
           kernel=factor(kernel, levels=c("<br>**Phylogeny**", "String", "UniFrac (UW)", "UniFrac (W)",
                                          "<br>**No Phylogeny**", "RBF", "Matern32")),
    )
}

# colour palette for kernels
# reds for phylogeny, blues for non-phylogeny
kernel_colour_palette <- c(
  setNames(
    colorRampPalette(brewer.pal(9,"Blues"))(4)[2:3],
    c("Matern32", "RBF")
  ),
  setNames(
    colorRampPalette(brewer.pal(9,"Reds"))(5)[4:2],
    c("String", "UniFrac (UW)", "UniFrac (W)")
  )
)
kernel_colour_palette[[ "Phylogeny" ]] <- "white"
kernel_colour_palette[[ "No Phylogeny" ]] <- "white"

####################################################################################
####################################################################################