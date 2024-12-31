library(data.table)
library(dplyr)

gpr_result <- "../results/ravel_gpr_model_evals.csv" %>%
  fread() %>%
  select(-phenotype) %>%
  rename(kernel=model) %>%
  pivot_longer(c(lml, lpd), names_to="quantity")


gpr_result %>%
  group_by(kernel, quantity) %>%
  summarise(value=median(value))

y_labels <- c(
  "gpr--lml"="Log-marginal likelihood",
  "gpr--lpd"="Log-predictive density"
)

plot_panels <- list(
  gpr=gpr_result
) %>%
  bind_rows(.id="model") %>%
  filter(kernel %in% c("linear", "string")) %>%
  mutate(kernel=recode(kernel, linear="Linear", string="String") %>%
           factor(levels=c("String", "Linear"))) %>%
  group_by(quantity, model) %>%
  do(
    plot=ggplot(data=., aes(x=kernel, y=value)) +
      geom_boxplot() +
      geom_point(size=3) +
      geom_line(aes(group=fold_idx)) +
      theme_linedraw() +
      theme(
        strip.background=element_blank(),
        strip.text=element_text(colour="black"),
        axis.text=element_text(size=46),
        axis.title=element_text(size=46),
        panel.grid.major.y=element_line(size=0.005, colour="grey50"),
        panel.grid.minor.y=element_line(size=0.005, colour="grey50"),
        panel.grid.major.x=element_blank()
      ) +
      xlab("Kernel") +
      ylab(y_labels[[ sprintf("%s--%s", .$model[[1]], .$quantity[[1]]) ]])
  )
plot_panels$plot[[1]]
plot_panels$plot[[2]]
