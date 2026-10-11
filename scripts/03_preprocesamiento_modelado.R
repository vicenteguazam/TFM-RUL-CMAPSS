
# ============================================================
# 03_preprocesamiento_modelado.R
#
# Análisis de las condiciones operativas y preparación
# inicial de las variables para la modelización
#
# Dataset: C-MAPSS FD002
# ============================================================


# ------------------------------------------------------------
# 1. LIBRERÍAS
# ------------------------------------------------------------

library(tidyverse)


# ------------------------------------------------------------
# 2. DIRECTORIO DE TRABAJO
# ------------------------------------------------------------

if (file.exists("scripts/train_FD002_prepared.csv")) {
  setwd("scripts")
} else if (!file.exists("train_FD002_prepared.csv")) {
  stop("No se encuentra train_FD002_prepared.csv. Ejecuta desde la raíz del repositorio o desde scripts/.")
}


# ------------------------------------------------------------
# 3. CARGA DEL CONJUNTO DE ENTRENAMIENTO
# ------------------------------------------------------------

train <- read_delim(
  "train_FD002_prepared.csv",
  delim = ";",
  show_col_types = FALSE
)


cat("\n============================================\n")
cat("PREPROCESAMIENTO PARA LA MODELIZACIÓN\n")
cat("============================================\n")

cat("Registros analizados:", nrow(train), "\n")
cat("Motores analizados:", n_distinct(train$unit_number), "\n")


# ------------------------------------------------------------
# 4. DEFINICIÓN DE VARIABLES
# ------------------------------------------------------------

settings <- c(
  "setting_1",
  "setting_2",
  "setting_3"
)

sensores <- paste0(
  "sensor_",
  1:21
)

variables_predictoras <- c(
  settings,
  sensores
)


cat("\n--- VARIABLES DEL MODELO ---\n")

cat(
  "Condiciones operativas:",
  length(settings),
  "\n"
)

cat(
  "Sensores:",
  length(sensores),
  "\n"
)

cat(
  "Variables predictoras iniciales:",
  length(variables_predictoras),
  "\n"
)

cat("Variable objetivo: RUL\n")
cat("Variables de identificación: unit_number y cycle\n")


# ------------------------------------------------------------
# 5. RESUMEN DE LAS VARIABLES OPERATIVAS
# ------------------------------------------------------------

cat("\n--- VARIABLES OPERATIVAS ---\n")

resumen_settings <- train %>%
  select(all_of(settings)) %>%
  summarise(
    across(
      everything(),
      list(
        minimo = ~ min(.x, na.rm = TRUE),
        maximo = ~ max(.x, na.rm = TRUE),
        media = ~ mean(.x, na.rm = TRUE),
        sd = ~ sd(.x, na.rm = TRUE),
        unicos = ~ n_distinct(.x)
      )
    )
  )

print(
  resumen_settings,
  width = Inf
)


# ------------------------------------------------------------
# 6. IDENTIFICACIÓN DE REGÍMENES OPERATIVOS
#
# setting_1 y setting_2 presentan pequeñas variaciones
# alrededor de valores nominales.
#
# Se realiza un redondeo para identificar las condiciones
# operativas nominales presentes en FD002.
# ------------------------------------------------------------

train_regimen <- train %>%
  mutate(
    setting_1_reg = round(setting_1, 0),
    setting_2_reg = round(setting_2, 2),
    setting_3_reg = round(setting_3, 0)
  )


regimenes <- train_regimen %>%
  count(
    setting_1_reg,
    setting_2_reg,
    setting_3_reg,
    name = "registros"
  ) %>%
  arrange(desc(registros)) %>%
  mutate(
    porcentaje = 100 * registros / sum(registros),
    regimen = row_number()
  ) %>%
  select(
    regimen,
    setting_1_reg,
    setting_2_reg,
    setting_3_reg,
    registros,
    porcentaje
  )


cat("\n--- REGÍMENES OPERATIVOS IDENTIFICADOS ---\n")

print(
  regimenes,
  n = Inf,
  width = Inf
)

cat(
  "\nNúmero de regímenes identificados:",
  nrow(regimenes),
  "\n"
)


# ------------------------------------------------------------
# 7. ASIGNACIÓN DEL RÉGIMEN A CADA OBSERVACIÓN
# ------------------------------------------------------------

train_regimen <- train_regimen %>%
  left_join(
    regimenes %>%
      select(
        regimen,
        setting_1_reg,
        setting_2_reg,
        setting_3_reg
      ),
    by = c(
      "setting_1_reg",
      "setting_2_reg",
      "setting_3_reg"
    )
  )


# ------------------------------------------------------------
# 8. DISTRIBUCIÓN DE LAS OBSERVACIONES POR RÉGIMEN
# ------------------------------------------------------------

distribucion_regimen <- train_regimen %>%
  count(
    regimen,
    name = "registros"
  ) %>%
  mutate(
    porcentaje = 100 * registros / sum(registros)
  ) %>%
  arrange(regimen)


cat("\n--- DISTRIBUCIÓN DE REGISTROS POR RÉGIMEN ---\n")

print(
  distribucion_regimen,
  n = Inf,
  width = Inf
)

cat(
  "\nRegistros sin régimen asignado:",
  sum(is.na(train_regimen$regimen)),
  "\n"
)


# ------------------------------------------------------------
# 9. NÚMERO DE REGÍMENES PRESENTES EN CADA MOTOR
# ------------------------------------------------------------

regimenes_por_motor <- train_regimen %>%
  group_by(unit_number) %>%
  summarise(
    numero_regimenes = n_distinct(regimen),
    .groups = "drop"
  )


distribucion_regimenes_motor <- regimenes_por_motor %>%
  count(
    numero_regimenes,
    name = "motores"
  ) %>%
  mutate(
    porcentaje = 100 * motores / sum(motores)
  ) %>%
  arrange(numero_regimenes)


cat("\n--- NÚMERO DE REGÍMENES PRESENTES POR MOTOR ---\n")

print(
  distribucion_regimenes_motor,
  n = Inf,
  width = Inf
)

cat(
  "\nMotores que presentan los 6 regímenes:",
  sum(regimenes_por_motor$numero_regimenes == 6),
  "de",
  nrow(regimenes_por_motor),
  "\n"
)


# ------------------------------------------------------------
# 10. ANÁLISIS DE LOS 21 SENSORES POR RÉGIMEN
#
# Para cada sensor y régimen se calcula:
#
# - media
# - desviación estándar
# - número de observaciones
# ------------------------------------------------------------

sensores_por_regimen <- train_regimen %>%
  select(
    regimen,
    all_of(sensores)
  ) %>%
  pivot_longer(
    cols = all_of(sensores),
    names_to = "sensor",
    values_to = "valor"
  ) %>%
  group_by(
    sensor,
    regimen
  ) %>%
  summarise(
    media = mean(valor, na.rm = TRUE),
    desviacion = sd(valor, na.rm = TRUE),
    observaciones = n(),
    .groups = "drop"
  )


cat("\n--- ANÁLISIS DE LOS 21 SENSORES POR RÉGIMEN ---\n")

print(
  sensores_por_regimen,
  n = Inf,
  width = Inf
)


# ------------------------------------------------------------
# 11. VARIACIÓN DE CADA SENSOR ENTRE REGÍMENES
#
# Se calcula:
#
# - media mínima entre regímenes
# - media máxima entre regímenes
# - rango entre medias
# - media global
# - variación relativa respecto a la media global
#
# La variación relativa es un indicador descriptivo.
# ------------------------------------------------------------

medias_globales <- train_regimen %>%
  select(all_of(sensores)) %>%
  pivot_longer(
    cols = everything(),
    names_to = "sensor",
    values_to = "valor"
  ) %>%
  group_by(sensor) %>%
  summarise(
    media_global = mean(valor, na.rm = TRUE),
    .groups = "drop"
  )


variacion_sensores <- sensores_por_regimen %>%
  group_by(sensor) %>%
  summarise(
    media_minima_regimen = min(media, na.rm = TRUE),
    media_maxima_regimen = max(media, na.rm = TRUE),
    rango_medias =
      media_maxima_regimen - media_minima_regimen,
    .groups = "drop"
  ) %>%
  left_join(
    medias_globales,
    by = "sensor"
  ) %>%
  mutate(
    variacion_relativa =
      if_else(
        abs(media_global) > .Machine$double.eps,
        100 * rango_medias / abs(media_global),
        NA_real_
      )
  ) %>%
  arrange(
    desc(variacion_relativa)
  )


cat("\n--- VARIACIÓN DE LOS SENSORES ENTRE REGÍMENES ---\n")

print(
  variacion_sensores,
  n = Inf,
  width = Inf
)


# ------------------------------------------------------------
# 12. CORRELACIÓN SENSOR-RUL DENTRO DE CADA RÉGIMEN
#
# En el EDA se calculó la correlación global entre cada
# variable y la RUL.
#
# Aquí se calcula la correlación de Pearson entre cada sensor
# y la RUL dentro de cada régimen operativo.
#
# Cuando un sensor presenta desviación estándar igual a cero
# dentro de un régimen, la correlación no está definida.
# En estos casos se asigna NA de forma explícita.
# ------------------------------------------------------------

correlacion_sensor_RUL_regimen <- train_regimen %>%
  select(
    regimen,
    RUL,
    all_of(sensores)
  ) %>%
  pivot_longer(
    cols = all_of(sensores),
    names_to = "sensor",
    values_to = "valor"
  ) %>%
  group_by(
    regimen,
    sensor
  ) %>%
  summarise(
    desviacion_sensor =
      sd(valor, na.rm = TRUE),
    
    correlacion_RUL =
      if (
        sd(valor, na.rm = TRUE) > 0 &&
        sd(RUL, na.rm = TRUE) > 0
      ) {
        cor(
          valor,
          RUL,
          use = "complete.obs",
          method = "pearson"
        )
      } else {
        NA_real_
      },
    
    .groups = "drop"
  ) %>%
  mutate(
    correlacion_absoluta =
      abs(correlacion_RUL)
  ) %>%
  arrange(
    regimen,
    desc(correlacion_absoluta)
  )


cat("\n--- CORRELACIÓN SENSOR-RUL POR RÉGIMEN ---\n")

print(
  correlacion_sensor_RUL_regimen,
  n = Inf,
  width = Inf
)


# ------------------------------------------------------------
# 13. RESUMEN DE CORRELACIÓN SENSOR-RUL
#
# Para cada sensor se calcula:
#
# - número de regímenes en los que puede calcularse Pearson
# - correlación media absoluta
# - correlación absoluta máxima
# - régimen en el que aparece la correlación máxima
#
# Los sensores constantes dentro de todos los regímenes
# mantienen NA como resultado.
# ------------------------------------------------------------

resumen_correlacion_regimen <- correlacion_sensor_RUL_regimen %>%
  group_by(sensor) %>%
  summarise(
    regimenes_con_correlacion =
      sum(!is.na(correlacion_RUL)),
    
    correlacion_media_absoluta =
      if (
        any(!is.na(correlacion_absoluta))
      ) {
        mean(
          correlacion_absoluta,
          na.rm = TRUE
        )
      } else {
        NA_real_
      },
    
    correlacion_maxima_absoluta =
      if (
        any(!is.na(correlacion_absoluta))
      ) {
        max(
          correlacion_absoluta,
          na.rm = TRUE
        )
      } else {
        NA_real_
      },
    
    regimen_correlacion_maxima =
      if (
        any(!is.na(correlacion_absoluta))
      ) {
        regimen[
          which.max(correlacion_absoluta)
        ]
      } else {
        NA_integer_
      },
    
    .groups = "drop"
  ) %>%
  arrange(
    desc(correlacion_media_absoluta)
  )


cat("\n--- RESUMEN DE CORRELACIÓN CON RUL POR SENSOR ---\n")

print(
  resumen_correlacion_regimen,
  n = Inf,
  width = Inf
)


# ------------------------------------------------------------
# 14. CREACIÓN DE CARPETA PARA FIGURAS
# ------------------------------------------------------------

if (!dir.exists("figuras_modelado")) {
  dir.create("figuras_modelado")
}


# ------------------------------------------------------------
# 15. FIGURA: DISTRIBUCIÓN DE LOS REGÍMENES
# ------------------------------------------------------------

grafico_regimenes <- ggplot(
  distribucion_regimen,
  aes(
    x = factor(regimen),
    y = registros
  )
) +
  geom_col() +
  labs(
    title = "Distribución de las condiciones operativas",
    x = "Régimen operativo",
    y = "Número de registros"
  ) +
  theme_minimal(base_size = 12)


print(grafico_regimenes)


ggsave(
  filename =
    "figuras_modelado/01_distribucion_regimenes.png",
  plot = grafico_regimenes,
  width = 8,
  height = 5,
  dpi = 300
)


# ------------------------------------------------------------
# 16. BOXPLOTS DE LOS SENSORES UTILIZADOS EN EL EDA
#
# Se representan los sensores 4, 11, 14 y 16 para mantener
# continuidad con el análisis exploratorio del apartado 3.5.
# ------------------------------------------------------------

sensores_graficos <- c(
  "sensor_4",
  "sensor_11",
  "sensor_14",
  "sensor_16"
)


for (sensor_actual in sensores_graficos) {
  
  grafico_sensor <- ggplot(
    train_regimen,
    aes(
      x = factor(regimen),
      y = .data[[sensor_actual]]
    )
  ) +
    geom_boxplot(
      outlier.alpha = 0.15
    ) +
    labs(
      title = paste(
        "Distribución de",
        sensor_actual,
        "según condición operativa"
      ),
      x = "Régimen operativo",
      y = sensor_actual
    ) +
    theme_minimal(base_size = 12)
  
  
  print(grafico_sensor)
  
  
  ggsave(
    filename = paste0(
      "figuras_modelado/02_",
      sensor_actual,
      "_regimenes.png"
    ),
    plot = grafico_sensor,
    width = 8,
    height = 5,
    dpi = 300
  )
}


# ------------------------------------------------------------
# 17. DATASET INICIAL PARA LA MODELIZACIÓN
#
# Se mantienen:
#
# - unit_number: identificación del motor
# - cycle: trazabilidad temporal
# - 3 condiciones operativas
# - 21 sensores
# - RUL
#
# unit_number y cycle NO se consideran inicialmente
# variables predictoras.
#
# El régimen tampoco se incorpora como predictor, puesto
# que se deriva directamente de los tres settings.
# ------------------------------------------------------------

datos_modelado <- train %>%
  select(
    unit_number,
    cycle,
    all_of(variables_predictoras),
    RUL
  )


cat("\n--- DATASET INICIAL PARA MODELIZACIÓN ---\n")

cat(
  "Registros:",
  nrow(datos_modelado),
  "\n"
)

cat(
  "Variables predictoras:",
  length(variables_predictoras),
  "\n"
)

cat(
  "Variables totales conservadas:",
  ncol(datos_modelado),
  "\n"
)


# ------------------------------------------------------------
# 18. EXPORTACIÓN DE RESULTADOS
# ------------------------------------------------------------

write_delim(
  datos_modelado,
  "train_FD002_modelado_inicial.csv",
  delim = ";"
)

write_delim(
  regimenes,
  "regimenes_operativos_FD002.csv",
  delim = ";"
)

write_delim(
  sensores_por_regimen,
  "sensores_por_regimen_FD002.csv",
  delim = ";"
)

write_delim(
  variacion_sensores,
  "variacion_sensores_regimen_FD002.csv",
  delim = ";"
)

write_delim(
  correlacion_sensor_RUL_regimen,
  "correlacion_sensor_RUL_por_regimen_FD002.csv",
  delim = ";"
)

write_delim(
  resumen_correlacion_regimen,
  "resumen_correlacion_RUL_regimen_FD002.csv",
  delim = ";"
)

write_delim(
  regimenes_por_motor,
  "regimenes_por_motor_FD002.csv",
  delim = ";"
)


# ------------------------------------------------------------
# 19. RESUMEN FINAL
# ------------------------------------------------------------

cat("\n============================================\n")
cat("RESUMEN DEL PREPROCESAMIENTO\n")
cat("============================================\n")

cat(
  "Registros analizados:",
  nrow(train_regimen),
  "\n"
)

cat(
  "Motores analizados:",
  n_distinct(train_regimen$unit_number),
  "\n"
)

cat(
  "Regímenes operativos identificados:",
  n_distinct(train_regimen$regimen),
  "\n"
)

cat(
  "Registros sin régimen asignado:",
  sum(is.na(train_regimen$regimen)),
  "\n"
)

cat(
  "Motores con los 6 regímenes:",
  sum(regimenes_por_motor$numero_regimenes == 6),
  "\n"
)

cat(
  "Sensores analizados:",
  length(sensores),
  "\n"
)

cat(
  "Variables predictoras iniciales:",
  length(variables_predictoras),
  "\n"
)

cat(
  "Dataset de modelización:",
  nrow(datos_modelado),
  "registros y",
  ncol(datos_modelado),
  "variables totales\n"
)

cat("============================================\n")
