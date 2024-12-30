library(ggplot2)
library(tibble)
library(data.table)
library(dplyr)

l_files <- list.files(
  file.path('..', 'results', 'gp_simulations', 'grid_search_test', 'regression', 'unifrac_kernel_model_selection'),
  full.names = TRUE
)

df_lml_grids <- lapply(l_files, fread) %>%
  rbindlist() %>%
  as_tibble()

format_lab <- function(x) {
  return(as.factor(round(x, 4)))
}

df_lml_grids %>%
  filter(phylo_spec == 'nonphylo', N_SAMPLES == 400) %>% 
  ggplot(aes(x=format_lab(noise_variance), y=format_lab(signal_variance))) +
  geom_tile(aes(fill=lml)) +
  facet_wrap(~rep) +
  theme(axis.text.x = element_text(angle=45, vjust=0.5))
  

df_lml_grids %>%
  select(phylo_spec, kernel, noise_variance, signal_variance) %>%
  distinct()

