
# ============================================================
# TFM - Estimación de la vida útil remanente (RUL)
# Dataset C-MAPSS - FD002
#
# 3.5 Análisis exploratorio de datos (EDA)
# ============================================================


# ------------------------------------------------------------
# 1. DIRECTORIO DE TRABAJO
# ------------------------------------------------------------

if (file.exists("scripts/train_FD002_prepared.csv")) {
  setwd("scripts")
} else if (!file.exists("train_FD002_prepared.csv")) {
  stop("No se encuentra train_FD002_prepared.csv. Ejecuta desde la raíz del repositorio o desde scripts/.")
}


# ------------------------------------------------------------
# 2. CARGA DE LIBRERÍAS
# ------------------------------------------------------------

library(tidyverse)


# ------------------------------------------------------------
# 3. IMPORTACIÓN DEL CONJUNTO DE ENTRENAMIENTO
# ------------------------------------------------------------

train <- read_delim(
  "train_FD002_prepared.csv",
  delim = ";",
  locale = locale(decimal_mark = "."),
  show_col_types = FALSE
)


# ------------------------------------------------------------
# 4. ESTADÍSTICA DESCRIPTIVA DE LA RUL
# ------------------------------------------------------------

cat("\n--- ESTADÍSTICA DESCRIPTIVA DE LA RUL ---\n")

print(summary(train$RUL))

cat("Media:",
    round(mean(train$RUL), 2), "\n")

cat("Desviación estándar:",
    round(sd(train$RUL), 2), "\n")


# ------------------------------------------------------------
# 5. ANÁLISIS DE LAS TRAYECTORIAS DE LOS MOTORES
# ------------------------------------------------------------

trayectorias <- train %>%
  group_by(unit_number) %>%
  summarise(
    ciclos = max(cycle),
    .groups = "drop"
  )

cat("\n--- LONGITUD DE LAS TRAYECTORIAS ---\n")

print(summary(trayectorias$ciclos))

cat("Trayectoria mínima:",
    min(trayectorias$ciclos), "ciclos\n")

cat("Trayectoria máxima:",
    max(trayectorias$ciclos), "ciclos\n")

cat("Trayectoria media:",
    round(mean(trayectorias$ciclos), 2), "ciclos\n")

cat("Desviación estándar:",
    round(sd(trayectorias$ciclos), 2), "ciclos\n")


# ------------------------------------------------------------
# 6. ESTADÍSTICA DESCRIPTIVA DE SETTINGS Y SENSORES
# ------------------------------------------------------------

variables_analisis <- train %>%
  select(
    setting_1:setting_3,
    sensor_1:sensor_21
  )

estadisticos_variables <- tibble(
  variable = names(variables_analisis),
  media = sapply(
    variables_analisis,
    mean
  ),
  desviacion = sapply(
    variables_analisis,
    sd
  ),
  minimo = sapply(
    variables_analisis,
    min
  ),
  maximo = sapply(
    variables_analisis,
    max
  ),
  valores_unicos = sapply(
    variables_analisis,
    n_distinct
  )
)

cat("\n--- ESTADÍSTICA DESCRIPTIVA DE LAS VARIABLES ---\n")

print(
  estadisticos_variables,
  n = Inf
)


# ------------------------------------------------------------
# 7. VARIABLES CONSTANTES O SIN VARIABILIDAD
# ------------------------------------------------------------

variables_constantes <- estadisticos_variables %>%
  filter(
    desviacion == 0 |
      valores_unicos == 1
  )

cat("\n--- VARIABLES CONSTANTES ---\n")

if (nrow(variables_constantes) == 0) {
  
  cat("No se han detectado variables constantes.\n")
  
} else {
  
  print(
    variables_constantes,
    n = Inf
  )
}


# ------------------------------------------------------------
# 8. CORRELACIÓN DE SETTINGS Y SENSORES CON LA RUL
# ------------------------------------------------------------

variables_correlacion <- train %>%
  select(
    setting_1:setting_3,
    sensor_1:sensor_21,
    RUL
  )

matriz_correlacion <- cor(
  variables_correlacion,
  use = "complete.obs",
  method = "pearson"
)

correlaciones_rul <- matriz_correlacion[, "RUL"]

tabla_correlaciones_rul <- tibble(
  variable = names(correlaciones_rul),
  correlacion_RUL = as.numeric(correlaciones_rul)
) %>%
  filter(
    variable != "RUL"
  ) %>%
  mutate(
    correlacion_absoluta = abs(correlacion_RUL)
  ) %>%
  arrange(
    desc(correlacion_absoluta)
  )

cat("\n--- CORRELACIÓN CON LA RUL ---\n")

print(
  tabla_correlaciones_rul,
  n = Inf
)


# ------------------------------------------------------------
# 9. CORRELACIÓN ENTRE SENSORES
# ------------------------------------------------------------

sensores <- train %>%
  select(sensor_1:sensor_21)

matriz_sensores <- cor(
  sensores,
  use = "complete.obs",
  method = "pearson"
)

# Extraer únicamente el triángulo superior de la matriz
# para evitar pares duplicados

indices <- which(
  abs(matriz_sensores) >= 0.90 &
    upper.tri(matriz_sensores),
  arr.ind = TRUE
)

if (nrow(indices) > 0) {
  
  pares_correlacion <- tibble(
    sensor_A = rownames(matriz_sensores)[indices[, 1]],
    sensor_B = colnames(matriz_sensores)[indices[, 2]],
    correlacion = matriz_sensores[indices]
  ) %>%
    arrange(desc(abs(correlacion)))
  
} else {
  
  pares_correlacion <- tibble(
    sensor_A = character(),
    sensor_B = character(),
    correlacion = numeric()
  )
}

cat("\n--- PARES DE SENSORES CON |r| >= 0.90 ---\n")

if (nrow(pares_correlacion) == 0) {
  
  cat("No se han detectado pares de sensores con |r| >= 0.90.\n")
  
} else {
  
  print(pares_correlacion)
}


# ------------------------------------------------------------
# 10. DETECCIÓN DESCRIPTIVA DE POSIBLES VALORES ATÍPICOS
#
# Criterio utilizado:
# valores inferiores a Q1 - 1.5*IQR
# valores superiores a Q3 + 1.5*IQR
#
# IMPORTANTE:
# En esta fase únicamente se identifican.
# No se elimina ninguna observación.
# ------------------------------------------------------------

contar_atipicos <- function(x) {
  
  q1 <- quantile(
    x,
    0.25,
    na.rm = TRUE
  )
  
  q3 <- quantile(
    x,
    0.75,
    na.rm = TRUE
  )
  
  rango_iqr <- IQR(
    x,
    na.rm = TRUE
  )
  
  limite_inferior <- q1 - 1.5 * rango_iqr
  limite_superior <- q3 + 1.5 * rango_iqr
  
  sum(
    x < limite_inferior |
      x > limite_superior,
    na.rm = TRUE
  )
}


tabla_atipicos <- tibble(
  variable = names(variables_analisis),
  
  numero_atipicos = sapply(
    variables_analisis,
    contar_atipicos
  )
) %>%
  
  mutate(
    porcentaje = round(
      100 * numero_atipicos / nrow(train),
      2
    )
  ) %>%
  
  arrange(
    desc(numero_atipicos)
  )


cat("\n--- POSIBLES VALORES ATÍPICOS ---\n")

print(
  tabla_atipicos,
  n = Inf
)


# ------------------------------------------------------------
# 11. RESUMEN DEL ANÁLISIS EXPLORATORIO
# ------------------------------------------------------------

cat("\n============================================\n")
cat("RESUMEN DEL ANÁLISIS EXPLORATORIO\n")
cat("============================================\n")

cat(
  "Registros analizados:",
  nrow(train),
  "\n"
)

cat(
  "Motores analizados:",
  n_distinct(train$unit_number),
  "\n"
)

cat(
  "RUL media:",
  round(mean(train$RUL), 2),
  "ciclos\n"
)

cat(
  "RUL mínima:",
  min(train$RUL),
  "ciclos\n"
)

cat(
  "RUL máxima:",
  max(train$RUL),
  "ciclos\n"
)

cat(
  "Trayectoria media:",
  round(mean(trayectorias$ciclos), 2),
  "ciclos\n"
)

cat(
  "Trayectoria mínima:",
  min(trayectorias$ciclos),
  "ciclos\n"
)

cat(
  "Trayectoria máxima:",
  max(trayectorias$ciclos),
  "ciclos\n"
)

cat(
  "Variables constantes:",
  nrow(variables_constantes),
  "\n"
)

cat(
  "Pares de sensores con |r| >= 0.90:",
  nrow(pares_correlacion),
  "\n"
)


# ============================================================
# 12. VISUALIZACIÓN DE LOS RESULTADOS
# ============================================================


# ------------------------------------------------------------
# 12.1 CREACIÓN DE CARPETA PARA LAS FIGURAS
# ------------------------------------------------------------

if (!dir.exists("figuras_EDA")) {
  dir.create("figuras_EDA")
}


# ------------------------------------------------------------
# 12.2 DISTRIBUCIÓN DE LA RUL
# ------------------------------------------------------------

grafico_rul <- ggplot(
  train,
  aes(x = RUL)
) +
  geom_histogram(
    bins = 30,
    color = "black"
  ) +
  labs(
    title = "Distribución de la vida útil remanente (RUL)",
    x = "RUL (ciclos)",
    y = "Frecuencia"
  ) +
  theme_minimal(base_size = 12)

print(grafico_rul)

ggsave(
  filename = "figuras_EDA/01_distribucion_RUL.png",
  plot = grafico_rul,
  width = 8,
  height = 5,
  dpi = 300
)


# ------------------------------------------------------------
# 12.3 DISTRIBUCIÓN DE LA DURACIÓN DE LAS TRAYECTORIAS
# ------------------------------------------------------------

grafico_trayectorias <- ggplot(
  trayectorias,
  aes(x = ciclos)
) +
  geom_histogram(
    bins = 20,
    color = "black"
  ) +
  labs(
    title = "Distribución de la duración de las trayectorias",
    x = "Duración de la trayectoria (ciclos)",
    y = "Número de motores"
  ) +
  theme_minimal(base_size = 12)

print(grafico_trayectorias)

ggsave(
  filename = "figuras_EDA/02_distribucion_trayectorias.png",
  plot = grafico_trayectorias,
  width = 8,
  height = 5,
  dpi = 300
)


# ------------------------------------------------------------
# 12.4 MATRIZ DE CORRELACIÓN ENTRE SENSORES
# ------------------------------------------------------------

matriz_cor_long <- as.data.frame(
  as.table(matriz_sensores)
)

names(matriz_cor_long) <- c(
  "Sensor_1",
  "Sensor_2",
  "Correlacion"
)

grafico_correlacion <- ggplot(
  matriz_cor_long,
  aes(
    x = Sensor_1,
    y = Sensor_2,
    fill = Correlacion
  )
) +
  geom_tile() +
  scale_fill_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  labs(
    title = "Matriz de correlación entre sensores",
    x = NULL,
    y = NULL,
    fill = "r"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    ),
    panel.grid = element_blank()
  )

print(grafico_correlacion)

ggsave(
  filename = "figuras_EDA/03_matriz_correlacion_sensores.png",
  plot = grafico_correlacion,
  width = 9,
  height = 8,
  dpi = 300
)


# ------------------------------------------------------------
# 12.5 CORRELACIÓN DE LAS VARIABLES CON LA RUL
# ------------------------------------------------------------

grafico_cor_rul <- tabla_correlaciones_rul %>%
  ggplot(
    aes(
      x = reorder(variable, correlacion_RUL),
      y = correlacion_RUL
    )
  ) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Correlación de las variables con la RUL",
    x = "Variable",
    y = "Coeficiente de correlación de Pearson"
  ) +
  theme_minimal(base_size = 12)

print(grafico_cor_rul)

ggsave(
  filename = "figuras_EDA/04_correlacion_variables_RUL.png",
  plot = grafico_cor_rul,
  width = 8,
  height = 7,
  dpi = 300
)


# ------------------------------------------------------------
# 12.6 EVOLUCIÓN PRELIMINAR DE SENSORES
#
# Se seleccionan algunos sensores que presentan diferentes
# comportamientos en el análisis previo. Esta figura se utiliza
# inicialmente como herramienta exploratoria.
# ------------------------------------------------------------

motores_ejemplo <- c(1, 2, 3)

sensores_ejemplo <- train %>%
  filter(
    unit_number %in% motores_ejemplo
  ) %>%
  select(
    unit_number,
    cycle,
    RUL,
    sensor_4,
    sensor_11,
    sensor_14,
    sensor_16
  ) %>%
  pivot_longer(
    cols = starts_with("sensor_"),
    names_to = "sensor",
    values_to = "valor"
  )

grafico_sensores <- ggplot(
  sensores_ejemplo,
  aes(
    x = cycle,
    y = valor,
    group = unit_number,
    color = factor(unit_number)
  )
) +
  geom_line(
    alpha = 0.8
  ) +
  facet_wrap(
    ~ sensor,
    scales = "free_y",
    ncol = 2
  ) +
  labs(
    title = "Evolución de sensores seleccionados",
    x = "Ciclo",
    y = "Valor del sensor",
    color = "Motor"
  ) +
  theme_minimal(base_size = 11)

print(grafico_sensores)

ggsave(
  filename = "figuras_EDA/05_evolucion_sensores_preliminar.png",
  plot = grafico_sensores,
  width = 10,
  height = 7,
  dpi = 300
)


# ------------------------------------------------------------
# 12.7 CONFIRMACIÓN
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FIGURAS DEL EDA GENERADAS CORRECTAMENTE\n")
cat("============================================\n")

cat(
  "Las figuras se han guardado en:",
  file.path(getwd(), "figuras_EDA"),
  "\n"
)
