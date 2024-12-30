library(dplyr)
library(data.table)
library(tidyr)
library(stringr)
library(ggplot2)
theme_set(theme_linedraw(base_size=12))
library(ggpubr)
library(ggforce)
library(ggh4x)
library(lemon)
library(cowplot)

source("../scripts/plotting/plot_utils.R")

#
# Read command line argument (path to GP host trait prediction simulation results)
args <- commandArgs(trailingOnly=TRUE)
save_path <- args[[1]]
save_path <- '../results/gp_simulations/manuscript.zip'
cat('Reading GP simulation results from ', save_path, '\n')

# if input is zipfile then unzip to temporary location and point script there
if(tools::file_ext(save_path) == 'zip') {
  cat('Input is a zip file - extracting to temporary location\n')
  tmp_dir <- file.path(tempdir(), 'gp_sim_plots')
  unzip(save_path, exdir=tmp_dir)
  save_path <- list.files(tmp_dir, full.names = TRUE)[[1]] # skip parent directory
}

#
# load results
save_paths <- c(
  file.path(save_path, 'classification'),
  file.path(save_path, 'regression')
)

pretty_kernel_names <- c(
  string="String",
  linear="Linear"
)

all_dirs <- lapply(
  save_paths,
  function(x) list.files(x, full.names=TRUE)
) %>%
  unlist(recursive=FALSE)

###############################################################################################################
# make the plots comparing the LML/ELBO and LPD between the string and linear kernels 
###############################################################################################################

model_evals <- all_dirs[ grepl("model_evals", basename(all_dirs)) ] %>%
  list.files(full.names=TRUE) %>%
  setNames(., .) %>%
  lapply(fread) %>%
  rbindlist(idcol="filename")
model_evals$TASK <- str_extract(model_evals$filename, 'reg|class')

for(plotted_quantity in c("lml", "lpd")) {
  plot_panels <- model_evals %>%
    mutate(
      facet_title=sprintf(
        "%s%s",
        c(reg="Regression", class="Classification")[TASK],
        if_else(
          TASK=="reg",
          paste0(",sigma_sq=", SIGMA_SQ),
          ""
        )
      ) %>%
        factor(
          levels=c("Regression,sigma_sq=0.3", "Regression,sigma_sq=0.6", "Classification"),
          labels=c(expression(atop("Regression", paste(sigma^2, "=0.3"))),
                   expression(atop("Regression", paste(sigma^2, "=0.6"))),
                   "Classification")
        )
    ) %>%
    filter(SAMPLE_READ_DISP==10) %>%
    select(-!!sym(setdiff(c("lml", "lpd"), plotted_quantity))) %>%
    mutate(phylo_spec=recode(as.character(phylo_spec),
                             "nonphylo"="Unrelated to phylogeny",
                             "phylo"="Related to phylogeny")
           ) %>%
    filter(model %in% c("linear", "string")) %>%
    pivot_wider(names_from=model, values_from=!!sym(plotted_quantity)) %>%
    group_by(facet_title) %>%
    do(
      plot=ggplot(
        data=.,
        aes(x=string, y=linear, colour=phylo_spec, shape=as.factor(N_SAMPLES))) +
        geom_point(size=3.5, stroke=0.75, fill="black") +
        facet_rep_grid(cols=vars(facet_title), labeller=label_parsed) +
        ylab("Linear kernel") +
        xlab("String kernel") +
        theme(
          panel.spacing.x=unit(2, "lines"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          panel.grid.minor.y=element_blank(),
          panel.grid.major.y=element_blank(),
          text=element_text(size=14),
          axis.title=element_text(size=18),
          axis.text=element_text(size=20),
          axis.line=element_line(),
          panel.border=element_blank(),
          strip.background=element_blank(),
          strip.text=element_text(colour="black", size=20),
          legend.position="top",
          legend.text=element_text(size=18),
          legend.title=element_text(size=20)
        ) +
        labs(colour="OTU effect sizes", shape="Dataset\nsize(n)") +
        geom_abline(linetype="dashed") +
        guides(
          colour=guide_legend(override.aes=list(size=5, shape=1)),
          shape=guide_legend(override.aes=list(size=5))
        ) +
        scale_shape_manual(values=c(21, 24))
    )
  
  plot_panels$plot <- lapply(plot_panels$plot, equalise_x_and_y_limits)
  
  fig <- ggarrange(
    plotlist=plot_panels$plot,
    common.legend=TRUE,
    legend="top",
    nrow=1,
    align="hv"
  )
  
  plot_filename <- sprintf(
    "../plots/supervised_learning__%s.pdf",
    plotted_quantity
  )
  
  cat('Saving plot to ', plot_filename, '\n')
  
  cropped_ggsave(
    plot_filename,
    plot=fig,
    width=12,
    height=5
  )
}

###############################################################################################################
###############################################################################################################



###############################################################################################################
# Plot of median LML/ELBO dependence on the string kernel hyperparameters
###############################################################################################################

named_file_list <- function(x) {
  out <- list.files(x, full.names=TRUE)
  names(out) <- out
  return(out)
}

string_lmls <- all_dirs[ grepl('string_lmls', all_dirs) ]

df_string_lmls <- lapply(
  string_lmls,
  function(x) lapply(named_file_list(x), fread) %>% rbindlist(idcol='filename')) %>%
  rbindlist() %>%
  as_tibble()
df_string_lmls$TASK <- str_extract(df_string_lmls$filename, 'reg|class')

df_median_string_lmls <- df_string_lmls %>%
  group_by_at(
    setdiff(colnames(.), c('lml', 'rep', 'RESAMPLE_BATCH_INDEX', 'filename'))
  ) %>%
  summarise(median_lml=median(lml), n_reps=n(), upper_lim=quantile(lml, 0.975), lower_lim=quantile(lml, 0.025)) %>%
  ungroup()

make_kernel_label <- function(type, m) {
  out <- rep(NA, length(type))
  out[ type == 'spectrum' ] <- 'Spectrum'
  out[ type == 'mismatch' ] <- paste0('Mismatch (m=', m[ type == 'mismatch' ], ')')
  out[ type == 'gappy' ] <- paste0('Gappy (g=', m[ type == 'gappy' ], ')')
  return(out)
}

unique_labels <- make_kernel_label(
  df_median_string_lmls$type,
  df_median_string_lmls$m
) %>% unique() %>% sort()

palette <- c(
  scales::brewer_pal('seq', 'Reds')(5+2) %>% tail(-2),
  scales::brewer_pal('seq', 'Blues')(5+2) %>% tail(-2),
  scales::brewer_pal('seq', 'Greens')(1+2) %>% tail(-2)
)
palette <- setNames(palette, unique_labels)

# subplot figures
reg_subplots <- df_median_string_lmls %>%
  filter(!(type=='mismatch' & m>=4)) %>%
  mutate(label=make_kernel_label(type, m)) %>%
  filter(TASK == 'reg', phylo_spec=='phylo', SAMPLE_READ_DISP == 10) %>%
  group_by(SIGMA_SQ, N_SAMPLES) %>%
  do(
    plot=ggplot(data=., aes(x=k, y=median_lml, group=label, colour=label)) +
      geom_line() +
      scale_colour_manual(values=palette, name='Kernel') +
      xlab('k-mer length') +
      ylab('Median LML over replicates') +
      scale_x_continuous(breaks=c(5, 10, 15, 20, 25, 30)) +
      theme(
        panel.spacing.x=unit(-10, "pt"),
        panel.spacing.y=unit(-20, "pt"),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.grid.minor.y=element_blank(),
        panel.grid.major.y=element_blank(),
        text=element_text(size=16),
        axis.title=element_text(size=12),
        axis.text=element_text(size=10),
        axis.line=element_line(),
        panel.border=element_blank(),
        strip.background=element_blank(),
        strip.text=element_text(colour="black", size=12)
      )
  )
plt <- reg_subplots$plot[[1]]

# reg_fig <- plot_grid(
#   ggpubr::get_legend(plt + guides(colour=guide_legend(ncol=5, override.aes = list(linewidth = 3)))),
#   plot_grid(
#     reg_subplots$plot[[1]] + theme(legend.position='none'),
#     reg_subplots$plot[[2]] + theme(legend.position='none'),
#     nrow=1
#   ),
#   ncol=1,
#   rel_heights = c(1, 2.5)
# )

reg_fig <- plot_grid(
  reg_subplots$plot[[1]] + theme(legend.position='none'),
  reg_subplots$plot[[2]] + theme(legend.position=c(0.7, 0.4), legend.title=element_blank()),
  nrow=1
)

plot_file_path <- file.path(
  '../plots',
  'regression_string_kernel_hparams_main_text.pdf'
)

cat('Saving plot to ', plot_file_path, '\n')

ggsave(
  plot_file_path,
  plot=reg_fig,
  height=4,
  width=10
)

class_subplots <- 
  df_median_string_lmls %>%
  mutate(label=make_kernel_label(type, m)) %>%
  filter(!(type=='mismatch' & m>=4)) %>%
  filter(TASK == 'class', phylo_spec=='phylo', SAMPLE_READ_DISP == 10) %>%
  group_by(N_SAMPLES) %>%
  do(
    plot=ggplot(data=., aes(x=k, y=median_lml, group=label, colour=label)) +
      geom_line() +
      scale_colour_manual(values=palette, name='Kernel') +
      scale_x_continuous(breaks=c(5, 10, 15, 20, 25, 30)) +
      xlab('k-mer length') +
      ylab('Median ELBO over replicates') +
      theme(
        panel.spacing.x=unit(-10, "pt"),
        panel.spacing.y=unit(-20, "pt"),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.grid.minor.y=element_blank(),
        panel.grid.major.y=element_blank(),
        text=element_text(size=16),
        axis.title=element_text(size=12),
        axis.text=element_text(size=10),
        axis.line=element_line(),
        panel.border=element_blank(),
        strip.background=element_blank(),
        strip.text=element_text(colour="black", size=12)
      )
  )

plt <- class_subplots$plot[[1]]

# class_fig <- plot_grid(
#   ggpubr::get_legend(plt + guides(colour=guide_legend(ncol=5, override.aes = list(linewidth = 3)))),
#   plot_grid(
#     class_subplots$plot[[1]] + theme(legend.position='none'),
#     class_subplots$plot[[2]] + theme(legend.position='none') + ylim(c(-222, -160)),
#     nrow=1
#   ),
#   ncol=1,
#   rel_heights = c(1, 2.5)
# )


class_fig <- plot_grid(
  class_subplots$plot[[1]] + theme(legend.position='none'),
  class_subplots$plot[[2]] + theme(legend.position=c(0.7, 0.4), legend.title=element_blank()) + ylim(c(-222, -160)),
  nrow=1
)

plot_file_path <- file.path(
  '../plots',
  'classification_string_kernel_hparams_main_text.pdf'
)

cat('Saving plot to ', plot_file_path, '\n')

ggsave(
  plot_file_path,
  plot=class_fig,
  height=4,
  width=10
)

###############################################################################################################
###############################################################################################################