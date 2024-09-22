library(dplyr)
library(data.table)
library(tidyr)
library(stringr)
library(ggplot2)
theme_set(theme_linedraw(base_size=12))
library(ggpubr)
library(ggforce)
# library(tidybayes)
# library(RColorBrewer)
# library(ggnewscale)
# library(ggtext)
library(ggh4x)
library(lemon)

source("../scripts/plotting/plot_utils.R")

#
# Read command line argument (path to MMD simulation results)
args <- commandArgs(trailingOnly=TRUE)
save_path <- args[[1]]
cat('Reading MMD simulation results from ', save_path, '\n')

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

# 
# #
# # supplemental - all kernel LMLs vs string
# dataset <- "fame__bacterial"
# plotted_quantity <- "lml"
# plt <- model_evals %>%
#   mutate(
#     facet_title=sprintf(
#       "%s%s",
#       c(reg="Regression", class="Classification")[TASK],
#       if_else(
#         TASK=="reg",
#         paste0("\nsigma_sq=", SIGMA_SQ),
#         ""
#       )
#     )
#   ) %>%
#   filter(DATASET==dataset, TRANSFORM==transform) %>%
#   select(-!!sym(setdiff(c("lml", "lpd"), plotted_quantity))) %>%
#   mutate(phylo_spec=recode(as.character(phylo_spec),
#                            "nonphylo"="Unrelated to phylogeny",
#                            "phylo"="Related to phylogeny")) %>%
#   pivot_wider(names_from=model, values_from=!!sym(plotted_quantity)) %>%
#   pivot_longer(c(linear, matern32, rbf), names_to="other_model") %>%
#   mutate(other_model=factor(pretty_kernel_names[ other_model ],
#                             levels=c("Linear", "RBF", "Matern32")),
#          x_label=sprintf("n=%d\nb=%s",
#                          N_SAMPLES, SAMPLE_READ_DISP)
#   ) %>%
#   arrange(N_SAMPLES, SIGMA_SQ, SAMPLE_READ_DISP) %>%
#   mutate(x_label=factor(x_label, levels=unique(x_label))) %>%
#   drop_na() %>%
#   group_by(facet_title) %>%
#   do(
#     plot=ggplot(
#       data=.,
#       aes(x=x_label, y=string-value, colour=phylo_spec)
#       ) +
#       geom_boxplot() +
#       facet_rep_grid(cols=vars(other_model), rows=vars(facet_title)) +
#       ylab(latex2exp::TeX(sprintf("$%s_{string} - %s_{other}$", toupper(plotted_quantity), toupper(plotted_quantity)))) +
#       xlab("Sample size (n) and total read dispersion (b)") +
#       geom_hline(yintercept=0, colour="red", size=1) +
#       theme(
#         panel.spacing.x=unit(-10, "pt"),
#         panel.spacing.y=unit(-20, "pt"),
#         panel.grid.major=element_blank(),
#         panel.grid.minor=element_blank(),
#         panel.grid.minor.y=element_blank(),
#         panel.grid.major.y=element_blank(),
#         text=element_text(size=16),
#         axis.title=element_text(size=12),
#         axis.text=element_text(size=10),
#         axis.line=element_line(),
#         panel.border=element_blank(),
#         strip.background=element_blank(),
#         strip.text=element_text(colour="black", size=12),
#         # legend.position=c(0.875, 0.25),
#         # legend.background=element_rect(colour="black")
#       ) +
#       labs(colour="OTU effect sizes") +
#       scale_y_continuous(breaks=scales::pretty_breaks())
#   ) %>%
#   pull(plot) %>%
#   ggarrange(plotlist=.,
#             ncol=1,
#             common.legend=TRUE)
# cropped_ggsave(
#   "../plots/fame_all_lmls.pdf",
#   plot=plt,
#   height=8, width=10
# )
# 
# #
# # supplemental - selected string kernel hparams in GP models
# pretty_string_kernel_names <- c(
#   gappy="Gappy pair",
#   mismatch="Mismatch",
#   spectrum="Spectrum"
# )
# pretty_facet_titles <- c(
#   "Spectrum ",
#   "Gappy pair (g=1)", "Gappy pair (g=2)", "Gappy pair (g=3)", "Gappy pair (g=4)", "Gappy pair (g=5)",
#   "Mismatch (g=1)", "Mismatch (g=2)", "Mismatch (g=3)", "Mismatch (g=4)", "Mismatch (g=5)"
# )
# 
# best_string_hparams <- all_dirs[ grepl("best_string_hparams", basename(all_dirs)) ] %>%
#   list.files(full.names=TRUE) %>%
#   setNames(., .) %>%
#   lapply(fread) %>%
#   rbindlist(idcol="filename") %>%
#   group_by(phylo_spec, DATASET, kernel_name, SIGMA_SQ, N_SAMPLES, SAMPLE_READ_DISP, TASK) %>%
#   tally() %>%
#   ungroup() %>%
#   filter(phylo_spec=="phylo") 
# 
# plot_panels <- best_string_hparams %>%
#   mutate(
#     subplot_name=sprintf(
#       "%s%s",
#       c(reg="Regression", class="Classification")[TASK],
#       if_else(
#         TASK=="reg",
#         paste0("\nsigma_sq=", SIGMA_SQ),
#         ""
#       )
#     )
#   ) %>%
#   filter(DATASET==dataset, SAMPLE_READ_DISP==10) %>%
#   separate(kernel_name, c("kernel_type", "k", "m_or_g"), remove=FALSE, sep=",") %>%
#   mutate(
#     k=as.integer(k),
#     m_or_g=ifelse(m_or_g=="NA", "", sprintf("(g=%s)", m_or_g)),
#     facet_title=sprintf("%s %s", pretty_string_kernel_names[ kernel_type ], m_or_g)
#   )  %>%
#   select(-c(SIGMA_SQ, SAMPLE_READ_DISP, kernel_name, kernel_type, m_or_g)) %>%
#   mutate(
#     facet_title=if_else(grepl("mismatch", facet_title), str_replace(facet_title, "g=", "m="), facet_title),
#     facet_title=factor(facet_title,
#                        levels=pretty_facet_titles,
#                        labels=if_else(grepl("Mismatch", pretty_facet_titles),
#                                       str_replace(pretty_facet_titles, "g", "m"),
#                                       pretty_facet_titles)
#     )
#   ) %>%
#   group_by(subplot_name) %>%
#   do(
#     plot=ggplot(
#       data=.,
#       aes(x=k, y=n, fill=as.factor(N_SAMPLES))) +
#       geom_col(position=position_dodge(preserve="single")) +
#       facet_rep_wrap(~facet_title, scales="free_x") +
#       xlab("k-mer length") +
#       ylab("Number of replciates") +
#       labs(fill="Sample size (n)") +
#       scale_x_continuous(breaks=seq(0, 30, by=5), limits=c(0, 30), expand=c(0,0)) +
#       scale_y_continuous(expand=c(0,0), limits=c(0, NA))+
#       theme(
#         panel.spacing.x=unit(-10, "pt"),
#         panel.grid.major=element_blank(),
#         panel.grid.minor=element_blank(),
#         panel.grid.minor.y=element_blank(),
#         panel.grid.major.y=element_blank(),
#         text=element_text(size=18),
#         axis.title=element_text(size=14),
#         axis.text=element_text(size=10),
#         axis.line=element_line(),
#         panel.border=element_blank(),
#         strip.background=element_blank(),
#         strip.text=element_text(colour="black", size=14)
#       )
#   ) 
# 
# for(i in 1:3) {
#   cropped_ggsave(
#     sprintf("../plots/best_string_hparams_%d.pdf", i),
#     plot=plot_panels$plot[[i]],
#     height=8, width=10
#   )
# }

