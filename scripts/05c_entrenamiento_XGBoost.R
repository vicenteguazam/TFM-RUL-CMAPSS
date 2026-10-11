
# ============================================================
# TFM - PREDICCIÓN DE RUL EN MOTORES TURBOFAN
# 05c_entrenamiento_XGBoost.R
#
# ENTRENAMIENTO DEL MODELO XGBOOST
#
# El modelo se entrena sobre la totalidad del conjunto TRAIN
# utilizando los hiperparámetros seleccionados previamente.
#
# El conjunto TEST no se utiliza en esta etapa.
# ============================================================


# ============================================================
# 1. LIBRERÍAS
# ============================================================

paquetes <- c(
  "tidyverse",
  "xgboost"
)

paquetes_faltantes <- paquetes[
  !paquetes %in% rownames(installed.packages())
]

if (length(paquetes_faltantes) > 0) {
  
  install.packages(
    paquetes_faltantes,
    dependencies = TRUE
  )
}

library(tidyverse)
library(xgboost)

cat("\nLibrerías cargadas correctamente.\n")


# ============================================================
# 2. RUTA DE TRABAJO
# ============================================================

if (file.exists("scripts/train_FD002_prepared.csv")) {
  setwd("scripts")
} else if (!file.exists("train_FD002_prepared.csv")) {
  stop("No se encuentra train_FD002_prepared.csv. Ejecuta desde la raíz del repositorio o desde scripts/.")
}

ruta <- "."

cat("\nDirectorio de trabajo:\n")
cat(getwd(), "\n")


# ============================================================
# 3. CARGA DE LOS DATOS PREPARADOS
# ============================================================

source(
  file.path(
    ruta,
    "04_preparacion_datos_modelado.R"
  )
)


cat("\n\n")
cat("============================================\n")
cat("ENTRENAMIENTO DEL MODELO XGBOOST\n")
cat("============================================\n")


# ============================================================
# 4. PREPARACIÓN DE LOS DATOS
# ============================================================

# XGBoost trabaja con matrices numéricas.

X_train_xgb <- as.matrix(X_train)

dtrain <- xgb.DMatrix(
  data = X_train_xgb,
  label = y_train
)


cat("\n--- DATOS DE ENTRENAMIENTO ---\n")

cat(
  "Observaciones:",
  nrow(X_train_xgb),
  "\n"
)

cat(
  "Variables predictoras:",
  ncol(X_train_xgb),
  "\n"
)

cat(
  "Motores:",
  n_distinct(train$unit_number),
  "\n"
)

cat("Variable objetivo: RUL\n")

cat(
  "Valores ausentes en predictores:",
  sum(is.na(X_train_xgb)),
  "\n"
)

cat(
  "Valores ausentes en RUL:",
  sum(is.na(y_train)),
  "\n"
)


# ============================================================
# 5. CONFIGURACIÓN DEL MODELO
# ============================================================

max_depth_xgb <- 5
eta_xgb <- 0.05
subsample_xgb <- 0.8
colsample_xgb <- 0.8
nrounds_xgb <- 300


cat("\n--- CONFIGURACIÓN XGBOOST ---\n")

cat(
  "max_depth:",
  max_depth_xgb,
  "\n"
)

cat(
  "eta:",
  eta_xgb,
  "\n"
)

cat(
  "subsample:",
  subsample_xgb,
  "\n"
)

cat(
  "colsample_bytree:",
  colsample_xgb,
  "\n"
)

cat(
  "nrounds:",
  nrounds_xgb,
  "\n"
)


# ============================================================
# 6. ENTRENAMIENTO DEL MODELO
# ============================================================

set.seed(123)

parametros_xgb <- list(
  
  objective = "reg:squarederror",
  
  max_depth = max_depth_xgb,
  
  eta = eta_xgb,
  
  subsample = subsample_xgb,
  
  colsample_bytree = colsample_xgb,
  
  eval_metric = "rmse"
)


tiempo_inicio <- Sys.time()


modelo_xgb <- xgb.train(
  
  params = parametros_xgb,
  
  data = dtrain,
  
  nrounds = nrounds_xgb,
  
  verbose = 0
)


tiempo_fin <- Sys.time()

tiempo_entrenamiento <- as.numeric(
  difftime(
    tiempo_fin,
    tiempo_inicio,
    units = "secs"
  )
)


cat("\nModelo entrenado correctamente.\n")

cat(
  "Tiempo de entrenamiento:",
  round(tiempo_entrenamiento, 2),
  "segundos\n"
)


# ============================================================
# 7. INFORMACIÓN DEL MODELO
# ============================================================

cat("\n--- INFORMACIÓN DEL MODELO ---\n")

cat(
  "Número de iteraciones:",
  nrounds_xgb,
  "\n"
)

cat(
  "Número de variables predictoras:",
  ncol(X_train_xgb),
  "\n"
)

cat(
  "Profundidad máxima:",
  max_depth_xgb,
  "\n"
)

cat(
  "Tasa de aprendizaje:",
  eta_xgb,
  "\n"
)


# ============================================================
# 8. IMPORTANCIA DE LAS VARIABLES
# ============================================================

importancia_xgb <- xgb.importance(
  feature_names = colnames(X_train_xgb),
  model = modelo_xgb
)

importancia_xgb <- as_tibble(
  importancia_xgb
)


cat("\n--- IMPORTANCIA DE VARIABLES ---\n")

print(
  importancia_xgb,
  n = nrow(importancia_xgb)
)


cat("\n--- 10 VARIABLES MÁS IMPORTANTES ---\n")

print(
  importancia_xgb %>%
    slice_head(n = 10)
)


# ============================================================
# 9. GRÁFICO DE IMPORTANCIA
# ============================================================

importancia_xgb_top10 <- importancia_xgb %>%
  
  slice_head(n = 10) %>%
  
  mutate(
    Feature = reorder(
      Feature,
      Gain
    )
  )


grafico_importancia_xgb <- ggplot(
  
  importancia_xgb_top10,
  
  aes(
    x = Gain,
    y = Feature
  )
  
) +
  
  geom_col() +
  
  labs(
    title = "Importancia de las variables en XGBoost",
    x = "Importancia (Gain)",
    y = "Variable"
  ) +
  
  theme_minimal()


print(grafico_importancia_xgb)


ggsave(
  
  filename = file.path(
    ruta,
    "importancia_variables_XGBoost.png"
  ),
  
  plot = grafico_importancia_xgb,
  
  width = 8,
  
  height = 5,
  
  dpi = 300
)


# ============================================================
# 10. GUARDADO DEL MODELO Y RESULTADOS
# ============================================================

saveRDS(
  
  modelo_xgb,
  
  file.path(
    ruta,
    "modelo_XGBoost_final.rds"
  )
)


write.csv(
  
  importancia_xgb,
  
  file.path(
    ruta,
    "importancia_variables_XGBoost.csv"
  ),
  
  row.names = FALSE
)


cat("\n--- ARCHIVOS GENERADOS ---\n")

cat("modelo_XGBoost_final.rds\n")
cat("importancia_variables_XGBoost.csv\n")
cat("importancia_variables_XGBoost.png\n")


# ============================================================
# 11. RESUMEN FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("RESUMEN DEL MODELO XGBOOST\n")
cat("============================================\n")

cat(
  "Observaciones utilizadas:",
  nrow(X_train_xgb),
  "\n"
)

cat(
  "Predictores:",
  ncol(X_train_xgb),
  "\n"
)

cat(
  "max_depth:",
  max_depth_xgb,
  "\n"
)

cat(
  "eta:",
  eta_xgb,
  "\n"
)

cat(
  "subsample:",
  subsample_xgb,
  "\n"
)

cat(
  "colsample_bytree:",
  colsample_xgb,
  "\n"
)

cat(
  "nrounds:",
  nrounds_xgb,
  "\n"
)

cat(
  "Tiempo de entrenamiento:",
  round(tiempo_entrenamiento, 2),
  "segundos\n"
)

cat(
  "TEST utilizado durante el entrenamiento: NO\n"
)


cat("\nVariable con mayor importancia:\n")

print(
  importancia_xgb %>%
    slice_head(n = 1)
)


cat("\n")
cat("============================================\n")
cat("ENTRENAMIENTO XGBOOST FINALIZADO\n")
cat("============================================\n")
