
# ============================================================
# TFM - PREDICCIÓN DE RUL EN MOTORES TURBOFAN
# 05a_ajuste_hiperparametros.R
#
# SELECCIÓN DE HIPERPARÁMETROS
#
# Modelos:
#   1. Random Forest
#   2. XGBoost
#   3. Red neuronal MLP
#
# La selección se realiza exclusivamente sobre TRAIN mediante
# validación cruzada de 5 folds agrupada por motor.
#
# Criterio de selección:
#   RMSE medio de validación cruzada.
#
# El conjunto TEST no se utiliza en este script.
# ============================================================


# ============================================================
# 1. LIBRERÍAS
# ============================================================

paquetes <- c(
  "tidyverse",
  "ranger",
  "xgboost",
  "nnet"
)

paquetes_faltantes <- paquetes[
  !paquetes %in% rownames(installed.packages())
]

if (length(paquetes_faltantes) > 0) {
  
  cat("\nInstalando paquetes necesarios:\n")
  print(paquetes_faltantes)
  
  install.packages(
    paquetes_faltantes,
    dependencies = TRUE
  )
}

library(tidyverse)
library(ranger)
library(xgboost)
library(nnet)

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
cat("AJUSTE DE HIPERPARÁMETROS\n")
cat("============================================\n")


# ============================================================
# 4. PREPARACIÓN DE LOS DATOS
# ============================================================

# Random Forest y XGBoost utilizan las variables
# en su escala original.

X_rf <- X_train
X_xgb <- as.matrix(X_train)

# La MLP utiliza las variables estandarizadas.

X_mlp <- as.matrix(X_train_scaled)

# Variable objetivo común a los tres modelos.

y <- y_train


cat("\n--- DATOS UTILIZADOS ---\n")

cat(
  "Observaciones:",
  nrow(X_train),
  "\n"
)

cat(
  "Variables predictoras:",
  ncol(X_train),
  "\n"
)

cat(
  "Motores:",
  n_distinct(train$unit_number),
  "\n"
)

cat(
  "Variable objetivo: RUL\n"
)

cat(
  "Conjunto utilizado: TRAIN\n"
)

cat(
  "TEST utilizado: NO\n"
)


# ============================================================
# 5. VALIDACIÓN CRUZADA AGRUPADA POR MOTOR
# ============================================================

# Se utilizan cinco folds.
#
# La asignación se realiza a nivel de motor para evitar que
# diferentes ciclos de una misma unidad aparezcan
# simultáneamente en entrenamiento y validación.

set.seed(123)

motores <- unique(train$unit_number)

motores <- sample(motores)

fold_motor <- rep(
  1:5,
  length.out = length(motores)
)

tabla_folds <- tibble(
  unit_number = motores,
  fold = fold_motor
)


datos_folds <- tibble(
  unit_number = train$unit_number
) %>%
  left_join(
    tabla_folds,
    by = "unit_number"
  )


cat("\n--- VALIDACIÓN CRUZADA ---\n")

cat("Número de folds: 5\n")
cat("Agrupación por motor: SI\n")

cat("\nMotores por fold:\n")

print(
  tabla_folds %>%
    count(fold)
)


# ============================================================
# 6. FUNCIÓN RMSE
# ============================================================

rmse <- function(real, predicho) {
  
  sqrt(
    mean(
      (real - predicho)^2
    )
  )
}


# ============================================================
# 7. RANDOM FOREST
# ============================================================

cat("\n")
cat("============================================\n")
cat("AJUSTE RANDOM FOREST\n")
cat("============================================\n")


# ------------------------------------------------------------
# 7.1 Valores candidatos
# ------------------------------------------------------------

# Se evalúa el hiperparámetro mtry, que determina el número
# de variables candidatas consideradas en cada división.

valores_mtry <- c(
  4,
  8,
  12,
  16
)

# Durante la selección de mtry se mantienen constantes:
#
# num.trees = 300
# min.node.size = 5

resultados_rf <- tibble()


# ------------------------------------------------------------
# 7.2 Validación cruzada
# ------------------------------------------------------------

for (mtry_actual in valores_mtry) {
  
  cat(
    "\nProbando mtry =",
    mtry_actual,
    "\n"
  )
  
  rmse_folds <- numeric(5)
  
  
  for (k in 1:5) {
    
    idx_train <- which(
      datos_folds$fold != k
    )
    
    idx_valid <- which(
      datos_folds$fold == k
    )
    
    
    datos_train_rf <- X_rf[idx_train, ] %>%
      mutate(
        RUL = y[idx_train]
      )
    
    
    modelo_rf_cv <- ranger(
      
      RUL ~ .,
      
      data = datos_train_rf,
      
      num.trees = 300,
      
      mtry = mtry_actual,
      
      min.node.size = 5,
      
      importance = "none",
      
      seed = 123
    )
    
    
    pred_rf <- predict(
      
      modelo_rf_cv,
      
      data = X_rf[idx_valid, ]
      
    )$predictions
    
    
    rmse_folds[k] <- rmse(
      y[idx_valid],
      pred_rf
    )
    
    
    cat(
      "Fold",
      k,
      "- RMSE:",
      round(rmse_folds[k], 4),
      "\n"
    )
  }
  
  
  resultados_rf <- bind_rows(
    
    resultados_rf,
    
    tibble(
      mtry = mtry_actual,
      RMSE_CV = mean(rmse_folds),
      SD_RMSE = sd(rmse_folds)
    )
  )
}


cat("\n--- RESULTADOS RANDOM FOREST ---\n")

print(
  resultados_rf %>%
    mutate(
      RMSE_CV = round(RMSE_CV, 4),
      SD_RMSE = round(SD_RMSE, 4)
    )
)


mejor_rf <- resultados_rf %>%
  slice_min(
    RMSE_CV,
    n = 1,
    with_ties = FALSE
  )


cat("\nConfiguración seleccionada:\n")

print(
  mejor_rf %>%
    mutate(
      RMSE_CV = round(RMSE_CV, 4),
      SD_RMSE = round(SD_RMSE, 4)
    )
)


# ============================================================
# 8. XGBOOST
# ============================================================

cat("\n")
cat("============================================\n")
cat("AJUSTE XGBOOST\n")
cat("============================================\n")


# ------------------------------------------------------------
# 8.1 Configuraciones candidatas
# ------------------------------------------------------------

# Se evalúan:
#
# max_depth -> profundidad máxima de los árboles.
# eta       -> tasa de aprendizaje.
#
# Se mantienen constantes:
#
# subsample = 0.8
# colsample_bytree = 0.8
# nrounds = 300

config_xgb <- tribble(
  
  ~configuracion, ~max_depth, ~eta,
  
  1,              3,          0.05,
  2,              5,          0.05,
  3,              7,          0.05,
  4,              3,          0.10,
  5,              5,          0.10,
  6,              7,          0.10
  
)

resultados_xgb <- tibble()


# ------------------------------------------------------------
# 8.2 Validación cruzada
# ------------------------------------------------------------

for (i in seq_len(nrow(config_xgb))) {
  
  cat(
    "\nConfiguración",
    config_xgb$configuracion[i],
    "- max_depth =",
    config_xgb$max_depth[i],
    "- eta =",
    config_xgb$eta[i],
    "\n"
  )
  
  
  rmse_folds <- numeric(5)
  
  
  for (k in 1:5) {
    
    idx_train <- which(
      datos_folds$fold != k
    )
    
    idx_valid <- which(
      datos_folds$fold == k
    )
    
    
    dtrain <- xgb.DMatrix(
      data = X_xgb[idx_train, ],
      label = y[idx_train]
    )
    
    
    dvalid <- xgb.DMatrix(
      data = X_xgb[idx_valid, ],
      label = y[idx_valid]
    )
    
    
    parametros <- list(
      
      objective = "reg:squarederror",
      
      eval_metric = "rmse",
      
      max_depth =
        config_xgb$max_depth[i],
      
      eta =
        config_xgb$eta[i],
      
      subsample = 0.8,
      
      colsample_bytree = 0.8
    )
    
    
    modelo_xgb_cv <- xgb.train(
      
      params = parametros,
      
      data = dtrain,
      
      nrounds = 300,
      
      verbose = 0
    )
    
    
    pred_xgb <- predict(
      modelo_xgb_cv,
      dvalid
    )
    
    
    rmse_folds[k] <- rmse(
      y[idx_valid],
      pred_xgb
    )
    
    
    cat(
      "Fold",
      k,
      "- RMSE:",
      round(rmse_folds[k], 4),
      "\n"
    )
  }
  
  
  resultados_xgb <- bind_rows(
    
    resultados_xgb,
    
    tibble(
      
      configuracion =
        config_xgb$configuracion[i],
      
      max_depth =
        config_xgb$max_depth[i],
      
      eta =
        config_xgb$eta[i],
      
      RMSE_CV =
        mean(rmse_folds),
      
      SD_RMSE =
        sd(rmse_folds)
    )
  )
}


cat("\n--- RESULTADOS XGBOOST ---\n")

print(
  resultados_xgb %>%
    mutate(
      RMSE_CV = round(RMSE_CV, 4),
      SD_RMSE = round(SD_RMSE, 4)
    )
)


mejor_xgb <- resultados_xgb %>%
  slice_min(
    RMSE_CV,
    n = 1,
    with_ties = FALSE
  )


cat("\nConfiguración seleccionada:\n")

print(
  mejor_xgb %>%
    mutate(
      RMSE_CV = round(RMSE_CV, 4),
      SD_RMSE = round(SD_RMSE, 4)
    )
)


# ============================================================
# 9. RED NEURONAL MLP
# ============================================================

cat("\n")
cat("============================================\n")
cat("AJUSTE RED NEURONAL MLP\n")
cat("============================================\n")


# ------------------------------------------------------------
# 9.1 Tamaños de capa oculta
# ------------------------------------------------------------

# Se utiliza una única capa oculta y se evalúan diferentes
# números de neuronas.

neuronas_candidatas <- c(
  15,
  20,
  25,
  30
)

# Se mantienen constantes:
#
# decay = 0.001
# maxit = 300
# linout = TRUE

resultados_mlp <- tibble()


# ------------------------------------------------------------
# 9.2 Validación cruzada
# ------------------------------------------------------------

for (neuronas in neuronas_candidatas) {
  
  cat(
    "\nProbando MLP con",
    neuronas,
    "neuronas ocultas\n"
  )
  
  
  rmse_folds <- numeric(5)
  
  
  for (k in 1:5) {
    
    idx_train <- which(
      datos_folds$fold != k
    )
    
    idx_valid <- which(
      datos_folds$fold == k
    )
    
    
    set.seed(123)
    
    
    modelo_mlp_cv <- nnet(
      
      x = X_mlp[idx_train, ],
      
      y = y[idx_train],
      
      size = neuronas,
      
      linout = TRUE,
      
      decay = 0.001,
      
      maxit = 300,
      
      trace = FALSE,
      
      MaxNWts = 10000
    )
    
    
    pred_mlp <- predict(
      
      modelo_mlp_cv,
      
      X_mlp[idx_valid, ]
    )
    
    
    rmse_folds[k] <- rmse(
      y[idx_valid],
      pred_mlp
    )
    
    
    cat(
      "Fold",
      k,
      "- RMSE:",
      round(rmse_folds[k], 4),
      "\n"
    )
  }
  
  
  resultados_mlp <- bind_rows(
    
    resultados_mlp,
    
    tibble(
      neuronas = neuronas,
      RMSE_CV = mean(rmse_folds),
      SD_RMSE = sd(rmse_folds)
    )
  )
}


cat("\n--- RESULTADOS MLP ---\n")

print(
  resultados_mlp %>%
    mutate(
      RMSE_CV = round(RMSE_CV, 4),
      SD_RMSE = round(SD_RMSE, 4)
    )
)


mejor_mlp_rmse <- resultados_mlp %>%
  slice_min(
    RMSE_CV,
    n = 1,
    with_ties = FALSE
  )


cat("\nConfiguración con menor RMSE:\n")

print(
  mejor_mlp_rmse %>%
    mutate(
      RMSE_CV = round(RMSE_CV, 4),
      SD_RMSE = round(SD_RMSE, 4)
    )
)


# ------------------------------------------------------------
# 9.3 Configuración seleccionada para la MLP
# ------------------------------------------------------------

# Cuando el aumento del número de neuronas no produce una
# mejora relevante del RMSE, se prioriza la arquitectura
# más sencilla para evitar complejidad innecesaria.

neuronas_mlp_seleccionadas <- 15


cat(
  "\nNúmero de neuronas seleccionado para el modelo final:",
  neuronas_mlp_seleccionadas,
  "\n"
)


# ============================================================
# 10. RESUMEN DE HIPERPARÁMETROS
# ============================================================

cat("\n")
cat("============================================\n")
cat("HIPERPARÁMETROS SELECCIONADOS\n")
cat("============================================\n")


cat("\nRANDOM FOREST\n")

cat(
  "mtry:",
  mejor_rf$mtry,
  "\n"
)

cat(
  "num.trees modelo final: 500\n"
)

cat(
  "min.node.size: 5\n"
)


cat("\nXGBOOST\n")

cat(
  "max_depth:",
  mejor_xgb$max_depth,
  "\n"
)

cat(
  "eta:",
  mejor_xgb$eta,
  "\n"
)

cat(
  "subsample: 0.8\n"
)

cat(
  "colsample_bytree: 0.8\n"
)

cat(
  "nrounds: 300\n"
)


cat("\nMLP\n")

cat(
  "Número de capas ocultas: 1\n"
)

cat(
  "Neuronas seleccionadas:",
  neuronas_mlp_seleccionadas,
  "\n"
)

cat(
  "decay: 0.001\n"
)

cat(
  "maxit modelo final: 500\n"
)


# ============================================================
# 11. GUARDADO DE LOS RESULTADOS DEL AJUSTE
# ============================================================

write.csv(
  resultados_rf,
  file.path(
    ruta,
    "ajuste_hiperparametros_RF.csv"
  ),
  row.names = FALSE
)


write.csv(
  resultados_xgb,
  file.path(
    ruta,
    "ajuste_hiperparametros_XGBoost.csv"
  ),
  row.names = FALSE
)


write.csv(
  resultados_mlp,
  file.path(
    ruta,
    "ajuste_hiperparametros_MLP.csv"
  ),
  row.names = FALSE
)


cat("\nResultados del ajuste guardados en archivos CSV.\n")


cat("\n")
cat("============================================\n")
cat("AJUSTE DE HIPERPARÁMETROS FINALIZADO\n")
cat("============================================\n")
