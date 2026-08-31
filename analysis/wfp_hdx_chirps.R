# reviewing CHIRPS
library(tidyverse)
library(gghdx)
library(cumulus)
library(glue)
library(grid)
gghdx()
adm_level <- 2
link <- "C:/Users/pauni/Downloads/eth-rainfall-subnat-full.csv"
rain_df <- read_csv(link, col_types = cols(r3q = col_double()), show_col_types = FALSE)[-1,] |>
  mutate(date = as.Date(date)) |>
  filter(date >= "2000-01-01")
rainfall_df <- rain_df |>
  filter(date == "2024-12-21", PCODE %in% sel_zones)
gdf <- cumulus::download_fieldmaps_sf(iso3, glue("{iso3}_adm{adm_level}"))[[1]]
gdf_adm1 <- cumulus::download_fieldmaps_sf(iso3, glue("{iso3}_adm1"))[[1]]
gdf |>
  left_join(
    rainfall_df,
    by = c("ADM2_PCODE" = "PCODE")
  ) |>
  ggplot() +
  geom_sf(aes(fill = r3q), color = NA) +
  geom_sf(data = gdf_adm1, fill = NA, color = "grey30", size = 0.75) +
  scale_fill_gradient2(
    low = "tomato", mid = "white", high = "steelblue",
    midpoint = 100, na.value = "grey95",
    limits = c(50, 250),
    breaks = seq(50, 250, by = 25)
  ) +
  labs(
    title = glue("Percent Rainfall Anomalies for Ethiopia, OND 2024 for ADM2"),
    fill = "Rainfall Anomaly\n(%)"
  ) + 
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    panel.background = element_blank()
  ) +
  guides(fill = guide_colorbar(barwidth = unit(180, "pt")))

sel_day <- rain_df %>%
  filter(PCODE %in% sel_zones,
         month(date) == 12, day(date) == 21)
year_means <- sel_day %>%
  mutate(year = year(date)) %>%
  group_by(year) %>%
  summarise(mean_agg = mean(r3q, na.rm = TRUE), .groups = "drop")

ggplot(year_means, aes(factor(year), mean_agg, fill = year == 2024)) +
  geom_col() +
  scale_fill_manual(values = c("grey80", "tomato"), guide = "none") +
  labs(title = "Mean Percent CHIRPS Rainfall Anomaly by Year",
       subtitle = "Ethiopia, OND",
    x = "Year", y = "Mean Percent Anomaly") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
