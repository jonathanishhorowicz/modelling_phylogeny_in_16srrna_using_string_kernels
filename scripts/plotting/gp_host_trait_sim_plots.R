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

source("../scripts/plotting/plot_utils.R")

#
# Read command line argument (path to MMD simulation results)
args <- commandArgs(trailingOnly=TRUE)
save_path <- args[[1]]
cat('Reading GP simulation results from ', save_path, '\n')

#
# load results
save_paths <- c(
  file.path(save_path, 'classification'),
  file.path(save_path, 'regression')
)

pretty_kernel_names <- c(
  rbf="RBF",
  string="String",
  linear="Linear",
  matern32="Matern32"
)

all_dirs <- lapply(
  save_paths,
  function(x) list.files(x, full.names=TRUE)
) %>%
  unlist(recursive=FALSE)

#
# make plots
model_evals <- all_dirs[ grepl("model_evals", basename(all_dirs)) ] %>%
  list.files(full.names=TRUE) %>%
  setNames(., .) %>%
  lapply(fread) %>%
  rbindlist(idcol="filename")

# main text - compare string and linear


# regression plot
transform <- "rel_abund"

# TODO: remove this - columns should be added in experiement script
model_evals$DATASET <- 'fame__bacterial'
model_evals$TASK <- str_extract(model_evals$filename, 'reg|class')



for(plotted_quantity in c("lml", "lpd")) {
  for(dataset in unique(model_evals$DATASET)) {
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
      filter(DATASET==dataset,
             TRANSFORM==transform,
             SAMPLE_READ_DISP==10) %>%
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
    
    cropped_ggsave(
      sprintf(
        "../plots/supervised_learning__%s_%s__%s.pdf",
        plotted_quantity,
        dataset,
        transform
      ),
      plot=fig,
      width=12,
      height=5
    )
  }
}
