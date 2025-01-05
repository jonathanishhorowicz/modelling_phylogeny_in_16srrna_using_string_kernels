library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
theme_set(theme_linedraw(base_size=12))

gpr_result <- "../results/ravel_gpr_model_evals.csv" %>%
  fread() %>%
  select(-phenotype) %>%
  rename(kernel=model) %>%
  pivot_longer(c(lml, lpd), names_to="quantity")

y_labels <- c(
  "lml"="Log-marginal likelihood",
  "lpd"="Log-predictive density"
)

plot_panels <- gpr_result %>%
  bind_rows(.id="model") %>%
  filter(kernel %in% c("linear", "string")) %>%
  mutate(
    kernel=recode(
      kernel,
      linear="Linear", string="String",
      ) %>%
      factor(levels=c("String", "Linear", 'RBF'))
  ) %>%
  group_by(quantity) %>%
  do(
    plot=ggplot(data=., aes(x=kernel, y=value)) +
      geom_boxplot() +
      geom_point(size=2) +
      geom_line(aes(group=fold_idx)) +
      theme_linedraw() +
      theme(
        panel.spacing.x=unit(-10, "pt"),
        panel.spacing.y=unit(-20, "pt"),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.grid.minor.y=element_blank(),
        panel.grid.major.y=element_blank(),
        text=element_text(size=16),
        axis.title=element_text(size=16),
        axis.text=element_text(size=14),
        axis.line=element_line(),
        panel.border=element_blank(),
        strip.background=element_blank(),
        strip.text=element_text(colour="black", size=12),
        # legend.position=c(0.875, 0.25),
        # legend.background=element_rect(colour="black")
      ) +
      xlab("Kernel") +
      ylab(y_labels[[ .$quantity[[1]] ]])
  )

# save both plots
lml_plot_save_path <- '../plots/ravel_gpr_lml.pdf'
cat('Saving plot to ', lml_plot_save_path, '\n')
ggsave(
  lml_plot_save_path,
  plot=plot_panels$plot[[1]],
  height=4,
  width=5
)

lpd_plot_save_path <- '../plots/ravel_gpr_lpd.pdf'
cat('Saving plot to ', lpd_plot_save_path, '\n')
ggsave(
  lpd_plot_save_path,
  plot=plot_panels$plot[[2]],
  height=4,
  width=5
)
