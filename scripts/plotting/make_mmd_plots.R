library(dplyr)
library(data.table)
library(tidyr)
library(tibble)
library(stringr)
library(forcats)
library(binom)

library(ggplot2)
theme_set(theme_linedraw(base_size=12))
library(ggpubr)
library(ggforce)
library(lemon)

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

get_t1err_confint <- function(nominal_sig, ci, n) {
  # Calculate the confidence interval around a nominal significance level using
  # the binomial proportion CI.
  df_ci <- binom.confint(nominal_sig * n, n, ci, method='exact')
  return(tibble(PointEst=nominal_sig, Lower=df_ci$lower, Upper=df_ci$upper))
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
  rbindlist()

# nominal significance level and its CI
alpha_level <- 0.1
alpha_level_interval_size <- 0.95

# H0 rejection rate
t1_errors  <- mmd_p_values %>%
  group_by_at(setdiff(colnames(.), c("SEED_CHUNK", "alpha_rep", "p_value", "mmd_emp"))) %>%
  summarise(n_reject=sum(p_value<alpha_level)/length(p_value))

# binomial proportion confidence intervals
t1_conf_int <- mmd_p_values %>%
  group_by_at(setdiff(colnames(mmd_p_values), c("SEED_CHUNK", "alpha_rep", "p_value", "mmd_emp"))) %>%
  summarise(get_t1err_confint(alpha_level, alpha_level_interval_size, n())) %>%
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

# main text plot
kernel_joiner <- tibble(
  facet_row_name=c("spectrum", "unifrac", "abundance_only"),
  k=list(c(30, 20, 10, 2), NA, NA),
  TRANSFORM=c("clr", "log1p", "log1p"),
  SAMPLE_READ_DISP="b=10"
) %>%
  unnest(k) %>%
  mutate(k=as.character(k))

plotdata2 <- t1_errors_plot %>%
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
  ) %>%
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
  ) %>%
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


make_h0_rejection_plot <- function(plot_data) {
  plot_data %>%
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
    get_plot_theme()
}

# Spectrum kernels
plotdata2 %>%
  filter(facet_row_name=="Spectrum") %>%
  mutate(k=factor(k, levels=plotdata2$k %>% as.integer() %>% unique() %>% sort())) %>%
  make_h0_rejection_plot() +
  labs(colour="k-mer length")

cropped_ggsave(
  "../plots/mmd_twosample_test_string.pdf",
  height=5,
  width=12
)

# UniFrac kernels
plotdata2 %>%
  filter(facet_row_name=="UniFrac") %>%
  mutate(kernel_name=recode(
    kernel_name,
    "unweighted-unifrac"="Unweighted",
    "weighted-unifrac"="Weighted")) %>%
  make_h0_rejection_plot() +
  labs(colour="UniFrac variant")
  
cropped_ggsave(
  "../plots/mmd_twosample_test_unifrac.pdf",
  height=5,
  width=12
)

# RBF and linear kernels
plotdata2 %>%
  filter(facet_row_name=="Abundance~only") %>%
  mutate(kernel_name=recode(
    kernel_name,
    "rbf-rescale"="RBF",
    "gram"="Linear")) %>%
  make_h0_rejection_plot() +
  labs(colour="Kernel type")
  
cropped_ggsave(
  "../plots/mmd_twosample_test_abundanceonly.pdf",
  height=5,
  width=12
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
  select(-c(p_value)) %>%
  pivot_wider(names_from=eps, values_from=mmd_emp) %>%
  group_by(kernel, N_TOTAL, SAMPLE_READ_DISP, TRANSFORM) %>%
  summarise(val=`0.1`/`1`) %>% 
  ggplot(aes(x=as.factor(N_TOTAL/2), y=val, colour=kernel)) +
  geom_boxplot(outlier.size=1) +
  labs(colour="Kernel", x="Group size") +
  ylab(expression(frac(MMD^2~(epsilon==0.1), MMD^2~(epsilon==1)))) +
  geom_hline(yintercept=1, colour="red") +
  coord_cartesian(ylim=c(0,1.5)) +
  get_plot_theme()
cropped_ggsave(
  "../plots/mmd_var_eps_ratio.pdf",
  width=12, height=4
)

#
# phylo vs random permutation
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
  filter(N_TOTAL==100,
          eps %in% c(0, 1e-3, 1e-2, 1e-1, 1)) %>%
  group_by(otu_perm_method, kernel, eps, N_TOTAL, SAMPLE_READ_DISP) %>%
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
  ylab(expression(MMD^2)) + xlab(expression(epsilon)) +
  labs(fill="OTU clustering", colour="OTU clustering") +
  ylim(c(0, NA))

cropped_ggsave(
  "../plots/mmd_for_different_eps.pdf",
  height=5, width=12
)