library(dplyr)
library(data.table)
library(tidyr)
library(tibble)
library(stringr)
library(forcats)
library(Hmisc) # TODO: remove this dependency, only used once and require installing ~10s other packgages

library(ggplot2)
theme_set(theme_linedraw(base_size=12))
library(ggpubr)
library(ggforce)
# library(tidybayes)
# library(RColorBrewer)
# library(ggnewscale)
# library(ggtext)
# library(ggh4x)
library(lemon)
# library(ggrepel)

source("../scripts/plotting/plot_utils.R")

########################################################################
# utility functions
########################################################################

get_kernel_type <- function(kernel) {
  key <- str_extract(
    kernel,
    "rbf|matern|spectrum|mismatch|gappy|gram|unifrac")
  prior_types <- c(
    matern="other",
    rbf="other",
    spectrum="string",
    mismatch="string",
    gappy="string",
    gram="other",
    unifrac="unifrac"
  )
  return(prior_types[ key ])
}

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

new_scale <- function(new_aes) {
  structure(ggplot2::standardise_aes_names(new_aes), class = "new_aes")
}

get_t1err_confint <- function(nominal_sig, ci, n) {
  # Calculate the confidence interval around a nominal significance level using the binomial CI.
  # alpha <- 1.0 - ci
  # nominal_sig + c(-qnorm(alpha),qnorm(alpha)) * sqrt((1/n) * nominal_sig * (1-nominal_sig))
  return(binconf(n*nominal_sig, n, alpha=1-ci) %>% as_tibble())
}

########################################################################
########################################################################

########################################################################
# main text plots
########################################################################

#
# Read command line argument (path to MMD simulation results)
args <- commandArgs(trailingOnly=TRUE)
save_path <- args[[1]]
cat('Reading MMD simulation results from ', save_path, '\n')

#
# Type I error plots
all_dirs <- list.files(save_path, full.names=TRUE)

mmd_p_value_files <- lapply(
  all_dirs[ grepl("mmd_and_pvalues", all_dirs) ],
  function(x) list.files(x, full.names=TRUE)
) %>% unlist()

mmd_p_values <- lapply(mmd_p_value_files, fread) %>%
  rbindlist() %>%
  filter(!kernel %in% c("rbf", "matern32"))

# nominal significance level
alpha_level <- 0.1

# H0 rejection rate
t1_errors  <- mmd_p_values %>%
  group_by_at(setdiff(colnames(.), c("SEED_CHUNK", "alpha_rep", "p_value", "mmd_emp"))) %>%
  # group_by(kernel, eps, DATASET, N_TOTAL, GROUP1_SIZE, otu_perm_method, SAMPLE_READ_DISP) %>%
  summarise(n_reject=sum(p_value<alpha_level)/length(p_value))

# binomial proportion confidence intervals
t1_conf_int <- mmd_p_values %>%
  group_by_at(setdiff(colnames(mmd_p_values), c("SEED_CHUNK", "alpha_rep", "p_value", "mmd_emp"))) %>%
  summarise(get_t1err_confint(alpha_level, 0.95, n())) %>%
  ungroup()

stopifnot(length(unique(t1_conf_int$Lower))==1)
stopifnot(length(unique(t1_conf_int$Upper))==1)
t1err_lower <- min(t1_conf_int$Lower)
t1err_upper <- max(t1_conf_int$Upper)

# Type I error for variable n
t1_errors_plot <- t1_errors %>%
  mutate(
    eps=ifelse(eps>1, 1, eps),
    eps=ifelse(eps<0, 0, eps),
    eps=factor(
      eps,
      levels=c(0, 1e-3, 1e-2, 1e-1, 1),
      labels=c(expression(paste(epsilon, "=0")),
               expression(paste(epsilon, "=1e-3")),
               expression(paste(epsilon, "=1e-2")),
               expression(paste(epsilon, "=1e-1")),
               expression(paste(epsilon, "=1"))
      )
    )
  ) %>%
  filter(!is.na(eps)) %>%
  mutate(kernel_type=get_kernel_type(kernel),
         legend_key=str_remove_all(
           kernel,
           "spectrum|mismatch|gappy|-unifrac|NA"
         ) %>% trimws("both", whitespace=",")) %>%
  separate(kernel, c("kernel_name", "k", "m"), sep=",", remove=FALSE)

# 
# string kernels with best hyperparameters

# main text plot
plot_1_data <- t1_errors_plot %>%
  filter(
    !grepl("-3", eps),
         otu_perm_method=="phylo",
         SAMPLE_READ_DISP!=1,
         !(grepl("unifrac", kernel) & TRANSFORM=="clr")) %>%
  mutate(
    kernel_name=kernel,
    SAMPLE_READ_DISP=factor(
      sprintf("b=%d", SAMPLE_READ_DISP)
    ),
    kernel_type=ifelse(
      grepl("gappy|spectrum|mismatch|unifrac", kernel),
      "Phylogenetic kernels",
      "Non-phylogenetic~kernels"
    )
  )

kernel_joiner <- tibble(
  facet_row_name=c("spectrum", "unifrac", "abundance_only"),
  k=list(c(30, 20, 10, 2), NA, NA),
  TRANSFORM=c("clr", "log1p", "log1p"),
  SAMPLE_READ_DISP="b=10"
) %>%
  unnest(k) %>%
  mutate(k=as.character(k))


# for(dataset_ in unique(plot_1_data$DATASET)) {

#   plot_1_data %>%
#     filter(DATASET==dataset_) %>%
#     mutate(
#       facet_row_name=str_extract(kernel, "spectrum|unifrac"),
#       facet_row_name=if_else(grepl("rbf|linear", kernel), "abundance_only", facet_row_name)
#     ) %>%
#     right_join(kernel_joiner) %>%
#     mutate(
#       facet_row_name=factor(
#         facet_row_name,
#         levels=c("spectrum", "unifrac", "abundance_only"),
#         labels=c("Spectrum", "UniFrac", "Abundance~only")
#       )
#     ) %>%
#     ggplot(aes(x=N_TOTAL/2, y=n_reject, colour=k)) +
#     geom_line(aes(group=kernel)) +
#     geom_point() +
#     facet_rep_grid(
#       rows=vars(facet_row_name),
#       cols=vars(eps),
#       labeller=label_parsed
#     ) +
#     geom_hline(yintercept=alpha_level,
#                colour="red",
#                size=0.5,
#                linetype="solid") +
#     geom_hline(yintercept=t1err_lower,
#                colour="red",
#                size=0.25,
#                linetype="dashed") +
#     geom_hline(yintercept=t1err_upper,
#                colour="red",
#                size=0.25,
#                linetype="dashed") +
#     scale_x_continuous(breaks=c(0, 100, 200), limits=c(0, 200)) +
#     scale_y_continuous(breaks=c(0, 0.1, 0.5, 1.0), limits=c(0, 1)) +
#     scale_color_manual(values=scales::hue_pal()(4), na.value="black") +
#     ylab(expression(paste(H[0], " rejection rate at nominal significance of 0.1"))) +
#     xlab("Group size") +
#     theme(
#       strip.background=element_blank(),
#       strip.text=element_text(size=12, colour="black"),
#       panel.grid.major=element_line(size=0.01, colour="grey50"),
#       panel.grid.minor=element_line(size=0.01, colour="grey50"),
#       axis.text=element_text(size=14),
#       axis.title=element_text(size=16),
#       panel.spacing.x=unit(-10, "pt"),
#       panel.spacing.y=unit(-5, "pt")
#     )
  
#   cropped_ggsave(
#     sprintf(
#       "../plots/mmd_H0_rejection_%s_%s.pdf",
#       dataset_,
#       str_replace_all(kernel_type_, " |~", "_")
#     ),
#     height=6,
#     width=11
#   )
# }

plotdata2 <-  plot_1_data %>%
  filter(DATASET=="fame__bacterial") %>%
  mutate(
    facet_row_name=str_extract(kernel, "spectrum|unifrac"),
    facet_row_name=if_else(grepl("rbf|gram", kernel), "abundance_only", facet_row_name)
  ) %>%
  right_join(kernel_joiner) %>%
  mutate(
    facet_row_name=factor(
      facet_row_name,
      levels=c("spectrum", "unifrac", "abundance_only"),
      labels=c("Spectrum", "UniFrac", "Abundance~only")
    )
  ) 

# spectrum kernels
get_plot_theme <- function() {
  theme(
    strip.background=element_blank(),
    strip.text=element_text(size=24, colour="black"),
    panel.grid.major=element_line(size=0.01, colour="grey50"),
    panel.grid.minor=element_line(size=0.01, colour="grey50"),
    legend.title=element_text(size=24),
    legend.text=element_text(size=20),
    axis.text=element_text(size=20),
    axis.title=element_text(size=24),
    panel.spacing.x=unit(-10, "pt"),
    panel.spacing.y=unit(-5, "pt")
  )
}
plotdata2 %>%
  filter(facet_row_name=="Spectrum") %>%
  mutate(k=factor(k, levels=plotdata2$k %>% as.integer() %>% unique() %>% sort())) %>%
  ggplot(aes(x=N_TOTAL/2, y=n_reject, colour=k)) +
  geom_line(aes(group=kernel)) +
  geom_point(size=3) +
  facet_rep_grid(
    cols=vars(eps),
    labeller=label_parsed
  ) +
  geom_hline(yintercept=alpha_level,
             colour="red",
             linewidth=0.5,
             linetype="solid") +
  geom_hline(yintercept=t1err_lower,
             colour="red",
             linewidth=0.25,
             linetype="dashed") +
  geom_hline(yintercept=t1err_upper,
             colour="red",
             linewidth=0.25,
             linetype="dashed") +
  scale_x_continuous(breaks=c(0, 100, 200), limits=c(0, 200)) +
  scale_y_continuous(breaks=c(0, 0.1, 0.5, 1.0), limits=c(0, 1)) +
  scale_color_manual(values=scales::hue_pal()(4), na.value="black") +
  ylab(expression(paste(H[0], " rejection rate"))) +
  xlab("Group size") +
  labs(colour="k-mer length") +
  get_plot_theme()
cropped_ggsave(
  "../plots/mmd_twosample_test_string.pdf"
)

# UniFrac
plotdata2 %>%
  filter(facet_row_name=="UniFrac") %>%
  mutate(kernel_name=recode(
    kernel_name,
    "unweighted-unifrac"="Unweighted",
    "weighted-unifrac"="Weighted")) %>%
  ggplot(aes(x=N_TOTAL/2, y=n_reject, colour=kernel_name)) +
  geom_line(aes(group=kernel)) +
  geom_point() +
  facet_rep_grid(
    cols=vars(eps),
    labeller=label_parsed
  ) +
  geom_hline(yintercept=alpha_level,
             colour="red",
             linewidth=0.5,
             linetype="solid") +
  geom_hline(yintercept=t1err_lower,
             colour="red",
             linewidth=0.25,
             linetype="dashed") +
  geom_hline(yintercept=t1err_upper,
             colour="red",
             linewidth=0.25,
             linetype="dashed") +
  scale_x_continuous(breaks=c(0, 100, 200), limits=c(0, 200)) +
  scale_y_continuous(breaks=c(0, 0.1, 0.5, 1.0), limits=c(0, 1)) +
  scale_color_manual(values=scales::hue_pal()(3) %>% tail(2), na.value="black") +
  ylab(expression(paste(H[0], " rejection rate"))) +
  xlab("Group size") +
  labs(colour="UniFrac variant") +
  get_plot_theme()
cropped_ggsave(
  "../plots/mmd_twosample_test_unifrac.pdf"
)

plotdata2 %>%
  filter(facet_row_name=="Abundance~only") %>%
  mutate(kernel_name=recode(
    kernel_name,
    "rbf-rescale"="RBF",
    "gram"="Linear")) %>%
  ggplot(aes(x=N_TOTAL/2, y=n_reject, colour=kernel_name)) +
  geom_line(aes(group=kernel)) +
  geom_point() +
  facet_rep_grid(
    cols=vars(eps),
    labeller=label_parsed
  ) +
  geom_hline(yintercept=alpha_level,
             colour="red",
             linewidth=0.5,
             linetype="solid") +
  geom_hline(yintercept=t1err_lower,
             colour="red",
             linewidth=0.25,
             linetype="dashed") +
  geom_hline(yintercept=t1err_upper,
             colour="red",
             linewidth=0.25,
             linetype="dashed") +
  scale_x_continuous(breaks=c(0, 100, 200), limits=c(0, 200)) +
  scale_y_continuous(breaks=c(0, 0.1, 0.5, 1.0), limits=c(0, 1)) +
  scale_color_manual(values=scales::hue_pal()(3) %>% tail(2), na.value="black") +
  ylab(expression(paste(H[0], " rejection rate"))) +
  xlab("Group size") +
  labs(colour="Kernel type") +
  get_plot_theme()
cropped_ggsave(
  "../plots/mmd_twosample_test_abundanceonly.pdf"
)


# supplementary text plot
hparam_name <- c(gappy="g=", mismatch="m=")

supp_fig_H0_plotdata <- t1_errors_plot %>%
  filter(!grepl("rbf|gram|matern", kernel)) %>%
  mutate(
    facet_title=sprintf("%s (%s%s)", kernel_name, hparam_name[kernel_name], m),
    facet_title=if_else(grepl("spectrum|unifrac", facet_title),
                        str_remove(facet_title, "\\(NANA\\)"),
                        facet_title),
    facet_title=str_remove(facet_title, "unweighted-|weighted-")
  ) 

plt_panel_data <- supp_fig_H0_plotdata %>%
  filter(SAMPLE_READ_DISP==10, otu_perm_method=="phylo") %>%
  right_join(
    tibble(
      kernel_name=c("gappy", "mismatch", "spectrum", "unweighted-unifrac", "weighted-unifrac"),
      TRANSFORM=c("clr", "clr", "clr", "log1p", "log1p")
    )
  ) %>%
  mutate(facet_title=trimws(facet_title)) %>%
  filter(facet_title!="unifrac") 

# for(dataset_name in c("fame", "Busselton")) {
  
  
#   fig <- plt_panel_data %>%
#     filter(!grepl("-3", eps), grepl(dataset_name, DATASET)) %>%
#     mutate(facet_title=str_replace_all(facet_title, " ", "\n")) %>%
#     ggplot(aes(x=as.integer(k), y=n_reject,
#         colour=as.factor(N_TOTAL/2))) +
#     geom_line(aes(group=N_TOTAL)) +
#     labs(colour="Group size", x="k-mer length") +
#     scale_y_continuous(limits=c(0,1), breaks=c(0, 0.5, 1.0)) +
#     scale_x_continuous(limits=c(2, 15)) +
#     facet_rep_grid(rows=vars(facet_title), cols=vars(eps),
#                    labeller=labeller(.rows=label_value, .cols=label_parsed)) +
#     theme(
#       strip.background=element_blank(),
#       strip.text=element_text(size=12, colour="black"),
#       panel.grid.major=element_line(size=0.01, colour="grey50"),
#       panel.grid.minor=element_line(size=0.01, colour="grey50"),
#       axis.text=element_text(size=10),
#       axis.title=element_text(size=12),
#       panel.spacing.x=unit(-10, "pt"),
#       panel.spacing.y=unit(-5, "pt")
#     ) +
#     geom_hline(yintercept=alpha_level,
#                colour="red",
#                size=0.5,
#                linetype="solid") +
#     geom_hline(yintercept=t1err_lower,
#                colour="red",
#                size=0.25,
#                linetype="dashed") +
#     geom_hline(yintercept=t1err_upper,
#                colour="red",
#                size=0.25,
#                linetype="dashed") +
#     ylab(expression(paste(H[0], " rejection rate at ", alpha, "=0.1")))
#   cropped_ggsave(
#     sprintf("../plots/H0_rejection_all_string__%s.pdf", dataset_name),
#     plot=fig,
#     height=12, width=10
#   )
# }

# plt_panel_data %>%
#   group_by(facet_title) %>%
#   do(
#     plot=ggplot(
#       data=.,
#       aes(x=as.integer(k), y=n_reject,
#           colour=as.factor(N_TOTAL/2),
#           linetype=eps)) +
#       geom_line(aes(group=interaction(N_TOTAL, eps))) +
#       labs(colour="Group size", linetype="b", x="k-mer length") +
#       scale_y_continuous(limits=c(0,1)) +
#       scale_x_continuous(limits=c(2, 15))
#       # ylab(expression(paste(H[0], " rejection rate\nat ", alpha, "=0.1")))
#   )
# plt_panels$plot[[1]]
# plt_panels <- setNames(plt_panels$plot, plt_panels$facet_title)
# 
# fig <- ggarrange(
#   plt_panels$`gappy (g=1)`, plt_panels$`gappy (g=2)`,
#   plt_panels$`gappy (g=3)`, plt_panels$`gappy (g=4)`,
#   plt_panels$`gappy (g=5)`, plt_panels$`mismatch (m=1)`,
#   plt_panels$`mismatch (m=2)`, plt_panels$`mismatch (m=3)`,
#   plt_panels$`mismatch (m=4)`, plt_panels$`mismatch (m=5)`,
#   ncol=2, nrow=5,
#   common.legend=TRUE
# )

# actual MMD values
#
mmd_plot_data <- mmd_p_values %>%
  mutate(
    eps=ifelse(eps>1, 1, eps),
    eps=ifelse(eps<0, 0, eps),
    eps=factor(
      eps,
      levels=c(0, 1e-3, 1e-2, 1e-1, 1),
      labels=c(expression(paste(epsilon, "=0")),
               expression(paste(epsilon, "=1e-3")),
               expression(paste(epsilon, "=1e-2")),
               expression(paste(epsilon, "=1e-1")),
               expression(paste(epsilon, "=1"))
      )
    )
  ) %>%
  filter(!is.na(eps)) %>%
  mutate(kernel_type=get_kernel_type(kernel)) %>%
  separate(kernel, c("kernel_name", "k", "m"), remove=FALSE) %>%
  filter(is.na(m) | m=="1" | m=="NA") 


for(dataset_ in unique(mmd_p_values$DATASET)) {
  mmd_p_values %>%
    right_join(
      tibble(
        kernel=c("spectrum,30,NA", "rbf-rescale", "gram", "unweighted-unifrac"),
      ) %>%
        mutate(TRANSFORM=if_else(grepl("unifrac", kernel), "log1p", "clr")),
      by=c("kernel", "TRANSFORM")
    ) %>%
    filter(otu_perm_method=="phylo",
           SAMPLE_READ_DISP==10) %>%
    mutate(
      eps=ifelse(eps>1, 1, eps),
      eps=ifelse(eps<0, 0, eps),
      TRANSFORM=factor(TRANSFORM,
                       levels=c("clr", "log1p"),
                       labels=c("clr(x)", "log(x+1)")),
      kernel=recode(
        kernel,
        gram="Linear",
        "rbf-rescale"="RBF",
        "spectrum,30,NA"="Spectrum (k=30)",
        "unweighted-unifrac"="Unweighted UniFrac"
      )
      ) %>%
    filter(DATASET==dataset_) %>%
    select(-c(p_value)) %>%
    pivot_wider(names_from=eps, values_from=mmd_emp) %>%
    group_by(kernel, N_TOTAL, SAMPLE_READ_DISP, TRANSFORM) %>%
    summarise(val=`0.1`/`1`) %>% 
    ggplot(aes(x=as.factor(N_TOTAL/2), y=val, colour=kernel)) +
    geom_boxplot(outlier.size=1) +
    # facet_rep_wrap(~kernel, nrow=1) +
    # facet_rep_grid(rows=vars(SAMPLE_READ_DISP), cols=vars(kernel)) + 
    labs(colour="Kernel", x="Group size") +
    ylab(expression(frac(MMD^2~(epsilon==0.1), MMD^2~(epsilon==1)))) +
    geom_hline(yintercept=1, colour="red") +
    coord_cartesian(ylim=c(0,1.5)) +
    get_plot_theme()
  cropped_ggsave(
    sprintf(
      "../plots/mmd_var_eps_ratio_%s.pdf",
      dataset_),
    width=12, height=4
  )
}

#
# phylo vs random permutation
for(dataset_ in unique(mmd_p_values$DATASET)) {
  
  mmd_p_values %>%
    right_join(
      tibble(
        kernel=c("spectrum,30,NA", "rbf-rescale", "gram", "unweighted-unifrac"),
      ) %>%
        mutate(TRANSFORM=if_else(grepl("unifrac", kernel), "log1p", "clr")),
      by=c("kernel", "TRANSFORM")
    ) %>%
    filter(SAMPLE_READ_DISP==10) %>%
    mutate(
      eps=ifelse(eps>1, 1, eps),
      eps=ifelse(eps<0, 0, eps),
      otu_perm_method=recode(otu_perm_method,
                             phylo="With phylogeny",
                             random="Without phylogeny"),
      kernel=recode(
        kernel,
        gram="Linear",
        "rbf-rescale"="RBF",
        "spectrum,30,NA"="Spectrum\n(k=30)",
        "unweighted-unifrac"="Unweighted\nUniFrac"
      )) %>%
    filter(DATASET==dataset_,
           N_TOTAL==100,
           eps %in% c(0, 1e-3, 1e-2, 1e-1, 1)) %>%
    group_by(DATASET, otu_perm_method, kernel, eps, N_TOTAL, SAMPLE_READ_DISP) %>%
    summarise(median_mmd=median(mmd_emp),
              upper_lim=quantile(mmd_emp, 0.975),
              lower_lim=quantile(mmd_emp, 0.025)) %>%
    mutate(eps=factor(eps,
                      levels=c(0, 1e-3, 1e-2, 1e-1, 1),
                      labels=format(c(0, 1e-3, 1e-2, 1e-1, 1), scienfitic=TRUE))) %>%
    ggplot(aes(x=eps, y=median_mmd,
               colour=otu_perm_method, group=interaction(otu_perm_method))) +
    geom_line() +
    geom_point() +
    geom_ribbon(aes(ymin=lower_lim, ymax=upper_lim,
                    fill=interaction(otu_perm_method)), alpha=0.5) +
    facet_rep_wrap(~kernel, scales="free_y", nrow=1) +
    get_plot_theme() +
    theme(
      axis.text.x=element_text(angle=60, hjust=1, vjust=1),
      legend.position="bottom",
      panel.spacing.x=unit(1, "lines")
    ) +
    # theme(
    #   panel.spacing.x=unit(1, "lines"),
    #   panel.grid.major=element_blank(),
    #   panel.grid.minor=element_blank(),
    #   panel.grid.minor.y=element_blank(),
    #   panel.grid.major.y=element_blank(),
    #   text=element_text(size=12),
    #   axis.text=element_text(size=10),
    #   axis.line=element_line(),
    #   panel.border=element_blank(),
    #   strip.background=element_blank(),
    #   strip.text=element_text(colour="black"),
    #   axis.text.x=element_text(angle=30, hjust=1, vjust=1)
    # ) +
    ylab(expression(MMD^2)) + xlab(expression(epsilon)) +
    labs(fill="OTU clustering", colour="OTU clustering") +
    ylim(c(0, NA))
  cropped_ggsave(sprintf(
    "../plots/mmd_for_different_eps_%s.pdf", dataset_),
    height=5, width=12)
}