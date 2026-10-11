
# ------------------------------------------------------------
# 4. CARGA DE LA RUL REAL DEL CONJUNTO TEST
# ------------------------------------------------------------

# El archivo RUL_FD002_prepared.csv contiene los valores
# reales de RUL correspondientes a los motores TEST.

rul_test <- read_delim(
  file.path(ruta, "RUL_FD002_prepared.csv"),
  delim = ";",
  show_col_types = FALSE
) %>%
  select(RUL_real)
