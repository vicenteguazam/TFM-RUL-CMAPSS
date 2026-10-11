
# ============================================================
# 07_analisis_errores_RF.R
# ANÁLISIS DEL ERROR DE PREDICCIÓN - RANDOM FOREST
# ============================================================

# ------------------------------------------------------------
# 1. LIBRERÍAS
# ------------------------------------------------------------

library(dplyr)
library(ggplot2)
library(tibble)
library(moments)

cat("============================================\n")
cat("ANÁLISIS DEL ERROR DE PREDICCIÓN - RANDOM FOREST\n")
cat("============================================\n\n")


# ------------------------------------------------------------
# 2. DIRECTORIO DE TRABAJO
# ------------------------------------------------------------

if (file.exists("scripts/train_FD002_prepared.csv")) {
  setwd("scripts")
} else if (!file.exists("train_FD002_prepared.csv")) {
  stop("No se encuentra train_FD002_prepared.csv. Ejecuta desde la raíz del repositorio o desde scripts/.")
}

ruta <- "."

cat("Directorio de trabajo:\n")
cat(getwd(), "\n\n")


# ------------------------------------------------------------
# 3. CARGA DE LOS RESULTADOS DEL CONJUNTO TEST
# ------------------------------------------------------------

predicciones <- read.csv(
  "predicciones_modelos_TEST.csv",
  stringsAsFactors = FALSE
)

cat("--- DATOS CARGADOS ---\n")
cat("Motores disponibles:", nrow(predicciones), "\n")
cat(
  "Variables disponibles:",
  paste(names(predicciones), collapse = ", "),
  "\n\n"
)


# ------------------------------------------------------------
# 4. COMPROBACIÓN Y CÁLCULO DEL ERROR
# ------------------------------------------------------------

# Si los errores no estuvieran ya almacenados, se calculan
if (!"error_RF" %in% names(predicciones)) {
  predicciones$error_RF <-
    predicciones$RUL_RF - predicciones$RUL_real
}

if (!"error_abs_RF" %in% names(predicciones)) {
  predicciones$error_abs_RF <-
    abs(predicciones$error_RF)
}

cat("--- COMPROBACIÓN ---\n")
cat("Motores analizados:", nrow(predicciones), "\n")
cat(
  "Valores ausentes:",
  sum(is.na(predicciones$RUL_real)) +
    sum(is.na(predicciones$RUL_RF)),
  "\n\n"
)


# ------------------------------------------------------------
# 5. ESTADÍSTICOS GLOBALES DEL ERROR
# ------------------------------------------------------------

error <- predicciones$error_RF
error_abs <- predicciones$error_abs_RF

media_error <- mean(error)
mediana_error <- median(error)
sd_error <- sd(error)

min_error <- min(error)
max_error <- max(error)

q1_error <- quantile(error, 0.25)
q3_error <- quantile(error, 0.75)
iqr_error <- IQR(error)

mae <- mean(error_abs)
rmse <- sqrt(mean(error^2))

p25_abs <- quantile(error_abs, 0.25)
p50_abs <- quantile(error_abs, 0.50)
p75_abs <- quantile(error_abs, 0.75)
p90_abs <- quantile(error_abs, 0.90)
p95_abs <- quantile(error_abs, 0.95)

cat("============================================\n")
cat("ESTADÍSTICOS GLOBALES DEL ERROR\n")
cat("============================================\n")

cat("Media del error:", round(media_error, 3), "ciclos\n")
cat("Mediana del error:", round(mediana_error, 3), "ciclos\n")
cat("Desviación estándar:", round(sd_error, 3), "ciclos\n")
cat("Error mínimo:", round(min_error, 3), "ciclos\n")
cat("Error máximo:", round(max_error, 3), "ciclos\n")
cat("Primer cuartil (Q1):", round(q1_error, 3), "ciclos\n")
cat("Tercer cuartil (Q3):", round(q3_error, 3), "ciclos\n")
cat("Rango intercuartílico:", round(iqr_error, 3), "ciclos\n")
cat("MAE:", round(mae, 3), "ciclos\n")
cat("RMSE:", round(rmse, 3), "ciclos\n\n")

cat("--- PERCENTILES DEL ERROR ABSOLUTO ---\n")
cat("P25:", round(p25_abs, 3), "ciclos\n")
cat("P50:", round(p50_abs, 3), "ciclos\n")
cat("P75:", round(p75_abs, 3), "ciclos\n")
cat("P90:", round(p90_abs, 3), "ciclos\n")
cat("P95:", round(p95_abs, 3), "ciclos\n\n")


# ------------------------------------------------------------
# 6. FORMA DE LA DISTRIBUCIÓN
# ------------------------------------------------------------

asimetria <- skewness(error)
curtosis <- kurtosis(error)

cat("============================================\n")
cat("FORMA DE LA DISTRIBUCIÓN DEL ERROR\n")
cat("============================================\n")

cat("Asimetría (skewness):", round(asimetria, 4), "\n")
cat("Curtosis:", round(curtosis, 4), "\n\n")


# ------------------------------------------------------------
# 7. TEST DE NORMALIDAD DE SHAPIRO-WILK
# ------------------------------------------------------------

shapiro_result <- shapiro.test(error)

cat("============================================\n")
cat("TEST DE NORMALIDAD DE SHAPIRO-WILK\n")
cat("============================================\n")

cat(
  "Estadístico W:",
  round(as.numeric(shapiro_result$statistic), 4),
  "\n"
)

cat(
  "p-valor:",
  format(shapiro_result$p.value, scientific = TRUE),
  "\n"
)

if (shapiro_result$p.value < 0.05) {
  cat(
    "Resultado: se rechaza la hipótesis de normalidad (p < 0.05).\n\n"
  )
} else {
  cat(
    "Resultado: no se rechaza la hipótesis de normalidad (p >= 0.05).\n\n"
  )
}


# ------------------------------------------------------------
# 8. SOBREESTIMACIÓN Y SUBESTIMACIÓN
# ------------------------------------------------------------

sobre_sub <- predicciones %>%
  mutate(
    tipo = ifelse(
      error_RF >= 0,
      "Sobreestimación",
      "Subestimación"
    )
  ) %>%
  count(tipo) %>%
  mutate(
    porcentaje = 100 * n / sum(n)
  )

cat("============================================\n")
cat("SOBREESTIMACIÓN Y SUBESTIMACIÓN\n")
cat("============================================\n")

print(sobre_sub)

cat("\n")


# ------------------------------------------------------------
# 9. ERROR SEGÚN INTERVALO DE RUL REAL
# ------------------------------------------------------------

predicciones <- predicciones %>%
  mutate(
    intervalo_RUL = cut(
      RUL_real,
      breaks = c(-Inf, 25, 50, 75, 100, 125, 150, Inf),
      labels = c(
        "≤25",
        "26-50",
        "51-75",
        "76-100",
        "101-125",
        "126-150",
        ">150"
      )
    )
  )

error_intervalos <- predicciones %>%
  group_by(intervalo_RUL) %>%
  summarise(
    motores = n(),
    RUL_media = mean(RUL_real),
    MAE = mean(abs(error_RF)),
    RMSE = sqrt(mean(error_RF^2)),
    sesgo = mean(error_RF),
    .groups = "drop"
  )

cat("============================================\n")
cat("ERROR SEGÚN INTERVALO DE RUL REAL\n")
cat("============================================\n")

print(error_intervalos)

cat("\n")


# ------------------------------------------------------------
# 10. MOTORES CON MAYOR ERROR ABSOLUTO
# ------------------------------------------------------------

mayores_errores <- predicciones %>%
  arrange(desc(error_abs_RF)) %>%
  select(
    motor,
    RUL_real,
    RUL_predicha = RUL_RF,
    error = error_RF,
    error_absoluto = error_abs_RF
  ) %>%
  slice_head(n = 10)

cat("============================================\n")
cat("10 MOTORES CON MAYOR ERROR ABSOLUTO\n")
cat("============================================\n")

print(mayores_errores)

cat("\n")


# ------------------------------------------------------------
# 11. GRÁFICO: ERROR SEGÚN RUL REAL
# ------------------------------------------------------------

grafico_error_rul <- ggplot(
  predicciones,
  aes(x = RUL_real, y = error_RF)
) +
  geom_point(alpha = 0.7) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  geom_smooth(
    method = "loess",
    se = FALSE
  ) +
  labs(
    title = "Error de predicción del Random Forest según la RUL real",
    x = "RUL real (ciclos)",
    y = "Error de predicción (ciclos)"
  ) +
  theme_minimal(base_size = 14)

ggsave(
  filename = file.path(ruta, "error_RF_segun_RUL.png"),
  plot = grafico_error_rul,
  width = 10,
  height = 6,
  dpi = 300,
  device = "png"
)


# ------------------------------------------------------------
# 12. HISTOGRAMA: DISTRIBUCIÓN DE LOS ERRORES
# ------------------------------------------------------------

grafico_distribucion <- ggplot(
  predicciones,
  aes(x = error_RF)
) +
  geom_histogram(
    bins = 25,
    boundary = 0
  ) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.8
  ) +
  labs(
    title = "Distribución de los errores de predicción del Random Forest",
    x = "Error de predicción (ciclos)",
    y = "Frecuencia"
  ) +
  theme_minimal(base_size = 14)

ggsave(
  filename = file.path(ruta, "distribucion_errores_RF.png"),
  plot = grafico_distribucion,
  width = 10,
  height = 6,
  dpi = 300,
  device = "png"
)


# ------------------------------------------------------------
# 13. GRÁFICO Q-Q
# ------------------------------------------------------------

grafico_qq <- ggplot(
  predicciones,
  aes(sample = error_RF)
) +
  stat_qq() +
  stat_qq_line() +
  labs(
    title = "Gráfico Q-Q de los errores del Random Forest",
    x = "Cuantiles teóricos de una distribución normal",
    y = "Cuantiles observados del error"
  ) +
  theme_minimal(base_size = 14)

ggsave(
  filename = file.path(ruta, "QQplot_errores_RF.png"),
  plot = grafico_qq,
  width = 8,
  height = 6,
  dpi = 300,
  device = "png"
)


# ------------------------------------------------------------
# 14. ANÁLISIS EXPLORATORIO DE CALIBRACIÓN
# ------------------------------------------------------------

cat("============================================\n")
cat("ANÁLISIS EXPLORATORIO DE CALIBRACIÓN\n")
cat("============================================\n\n")


# ------------------------------------------------------------
# 14.1 CALIBRACIÓN LINEAL
# ------------------------------------------------------------

modelo_cal_lineal <- lm(
  RUL_real ~ RUL_RF,
  data = predicciones
)

summary_lineal <- summary(modelo_cal_lineal)

intercepto_lineal <- coef(modelo_cal_lineal)[1]
pendiente_lineal <- coef(modelo_cal_lineal)[2]

r2_lineal <- summary_lineal$r.squared
r2_adj_lineal <- summary_lineal$adj.r.squared

rmse_res_lineal <- sqrt(
  mean(residuals(modelo_cal_lineal)^2)
)

aic_lineal <- AIC(modelo_cal_lineal)

cat("--- CALIBRACIÓN LINEAL ---\n")
cat(
  "Intercepto:",
  round(intercepto_lineal, 4),
  "\n"
)
cat(
  "Pendiente:",
  round(pendiente_lineal, 4),
  "\n"
)
cat(
  "R2:",
  round(r2_lineal, 4),
  "\n"
)
cat(
  "R2 ajustado:",
  round(r2_adj_lineal, 4),
  "\n"
)
cat(
  "RMSE residual:",
  round(rmse_res_lineal, 3),
  "ciclos\n\n"
)


# ------------------------------------------------------------
# 14.2 CALIBRACIÓN CUADRÁTICA
# ------------------------------------------------------------

modelo_cal_cuadratico <- lm(
  RUL_real ~ RUL_RF + I(RUL_RF^2),
  data = predicciones
)

summary_cuadratico <- summary(modelo_cal_cuadratico)

coef_cuadratico <- coef(modelo_cal_cuadratico)

r2_cuadratico <- summary_cuadratico$r.squared
r2_adj_cuadratico <- summary_cuadratico$adj.r.squared

rmse_res_cuadratico <- sqrt(
  mean(residuals(modelo_cal_cuadratico)^2)
)

aic_cuadratico <- AIC(modelo_cal_cuadratico)

cat("--- CALIBRACIÓN CUADRÁTICA ---\n")
cat(
  "Intercepto:",
  round(coef_cuadratico[1], 4),
  "\n"
)
cat(
  "Coeficiente lineal:",
  round(coef_cuadratico[2], 4),
  "\n"
)
cat(
  "Coeficiente cuadrático:",
  round(coef_cuadratico[3], 6),
  "\n"
)
cat(
  "R2:",
  round(r2_cuadratico, 4),
  "\n"
)
cat(
  "R2 ajustado:",
  round(r2_adj_cuadratico, 4),
  "\n"
)
cat(
  "RMSE residual:",
  round(rmse_res_cuadratico, 3),
  "ciclos\n\n"
)


# ------------------------------------------------------------
# 15. COMPARACIÓN DE LOS MODELOS DE CALIBRACIÓN
# ------------------------------------------------------------

comparacion_calibracion <- tibble(
  modelo = c(
    "Calibración lineal",
    "Calibración cuadrática"
  ),
  R2 = c(
    r2_lineal,
    r2_cuadratico
  ),
  R2_ajustado = c(
    r2_adj_lineal,
    r2_adj_cuadratico
  ),
  RMSE_residual = c(
    rmse_res_lineal,
    rmse_res_cuadratico
  ),
  AIC = c(
    aic_lineal,
    aic_cuadratico
  )
)

cat("============================================\n")
cat("COMPARACIÓN DE MODELOS DE CALIBRACIÓN\n")
cat("============================================\n")

print(comparacion_calibracion)

cat("\n")


# ------------------------------------------------------------
# 16. COMPARACIÓN ANOVA
# ------------------------------------------------------------

anova_calibracion <- anova(
  modelo_cal_lineal,
  modelo_cal_cuadratico
)

cat("============================================\n")
cat("COMPARACIÓN ANOVA: LINEAL VS CUADRÁTICO\n")
cat("============================================\n")

print(anova_calibracion)

cat("\n")


# ------------------------------------------------------------
# 17. GRÁFICO DE CALIBRACIÓN LINEAL
# ------------------------------------------------------------

grafico_cal_lineal <- ggplot(
  predicciones,
  aes(x = RUL_RF, y = RUL_real)
) +
  geom_point(alpha = 0.7) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = FALSE
  ) +
  labs(
    title = "Calibración del modelo Random Forest",
    subtitle = "RUL real frente a RUL predicha",
    x = "RUL predicha (ciclos)",
    y = "RUL real (ciclos)"
  ) +
  theme_minimal(base_size = 14)

ggsave(
  filename = file.path(ruta, "calibracion_RF_lineal.png"),
  plot = grafico_cal_lineal,
  width = 8,
  height = 7,
  dpi = 300,
  device = "png"
)


# ------------------------------------------------------------
# 18. GRÁFICO DE CALIBRACIÓN CUADRÁTICA
# ------------------------------------------------------------

grafico_cal_cuadratica <- ggplot(
  predicciones,
  aes(x = RUL_RF, y = RUL_real)
) +
  geom_point(alpha = 0.7) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +
  geom_smooth(
    method = "lm",
    formula = y ~ x + I(x^2),
    se = FALSE
  ) +
  labs(
    title = "Análisis exploratorio de calibración del Random Forest",
    subtitle = "Ajuste cuadrático frente a la calibración ideal",
    x = "RUL predicha (ciclos)",
    y = "RUL real (ciclos)"
  ) +
  theme_minimal(base_size = 14)

ggsave(
  filename = file.path(ruta, "calibracion_RF_cuadratica.png"),
  plot = grafico_cal_cuadratica,
  width = 8,
  height = 7,
  dpi = 300,
  device = "png"
)


# ------------------------------------------------------------
# 19. TABLA DE ESTADÍSTICOS DE LA DISTRIBUCIÓN
# ------------------------------------------------------------

estadisticos_distribucion <- tibble(
  estadistico = c(
    "Media",
    "Mediana",
    "Desviación estándar",
    "Mínimo",
    "Máximo",
    "Q1",
    "Q3",
    "Rango intercuartílico",
    "Asimetría",
    "Curtosis",
    "Shapiro-Wilk W",
    "Shapiro-Wilk p"
  ),
  valor = c(
    media_error,
    mediana_error,
    sd_error,
    min_error,
    max_error,
    as.numeric(q1_error),
    as.numeric(q3_error),
    iqr_error,
    asimetria,
    curtosis,
    as.numeric(shapiro_result$statistic),
    shapiro_result$p.value
  )
)


# ------------------------------------------------------------
# 20. GUARDADO DE RESULTADOS
# ------------------------------------------------------------

write.csv(
  predicciones,
  "errores_prediccion_RF.csv",
  row.names = FALSE
)

write.csv(
  error_intervalos,
  "error_RF_por_intervalo_RUL.csv",
  row.names = FALSE
)

write.csv(
  mayores_errores,
  "mayores_errores_RF.csv",
  row.names = FALSE
)

write.csv(
  sobre_sub,
  "sobreestimacion_subestimacion_RF.csv",
  row.names = FALSE
)

write.csv(
  estadisticos_distribucion,
  "estadisticos_distribucion_error_RF.csv",
  row.names = FALSE
)

write.csv(
  comparacion_calibracion,
  "comparacion_calibracion_RF.csv",
  row.names = FALSE
)


# ------------------------------------------------------------
# 21. RESUMEN FINAL
# ------------------------------------------------------------

cat("============================================\n")
cat("RESUMEN DEL ANÁLISIS DE ERRORES\n")
cat("============================================\n")

cat("Motores analizados:", nrow(predicciones), "\n")
cat("MAE:", round(mae, 3), "ciclos\n")
cat("RMSE:", round(rmse, 3), "ciclos\n")
cat("Media del error:", round(media_error, 3), "ciclos\n")
cat("Mediana del error:", round(mediana_error, 3), "ciclos\n")
cat("Desviación estándar:", round(sd_error, 3), "ciclos\n")
cat("Asimetría:", round(asimetria, 4), "\n")
cat("Curtosis:", round(curtosis, 4), "\n")
cat(
  "Shapiro-Wilk p-valor:",
  format(shapiro_result$p.value, scientific = TRUE),
  "\n\n"
)

cat("Calibración lineal:\n")
cat(
  "Intercepto:",
  round(intercepto_lineal, 4),
  "\n"
)
cat(
  "Pendiente:",
  round(pendiente_lineal, 4),
  "\n"
)
cat(
  "R2:",
  round(r2_lineal, 4),
  "\n"
)
cat(
  "RMSE residual:",
  round(rmse_res_lineal, 3),
  "ciclos\n\n"
)

cat("Calibración cuadrática:\n")
cat(
  "R2:",
  round(r2_cuadratico, 4),
  "\n"
)
cat(
  "RMSE residual:",
  round(rmse_res_cuadratico, 3),
  "ciclos\n\n"
)

cat("Archivos generados:\n")
cat("errores_prediccion_RF.csv\n")
cat("error_RF_por_intervalo_RUL.csv\n")
cat("mayores_errores_RF.csv\n")
cat("sobreestimacion_subestimacion_RF.csv\n")
cat("estadisticos_distribucion_error_RF.csv\n")
cat("comparacion_calibracion_RF.csv\n")
cat("error_RF_segun_RUL.png\n")
cat("distribucion_errores_RF.png\n")
cat("QQplot_errores_RF.png\n")
cat("calibracion_RF_lineal.png\n")
cat("calibracion_RF_cuadratica.png\n\n")

cat("============================================\n")
cat("ANÁLISIS DE ERRORES FINALIZADO\n")
cat("============================================\n")
