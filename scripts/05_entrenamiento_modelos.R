
# ============================================================
# TFM - PREDICCIÓN DE RUL EN MOTORES TURBOFAN
# 05_entrenamiento_modelos.R
#
# Apartado 4.5 - Entrenamiento y ajuste del modelo
#
# 4.5.1 Preparación de las variables de entrada
# 4.5.2 Entrenamiento del modelo Random Forest
# 4.5.3 Entrenamiento del modelo XGBoost
# 4.5.4 Entrenamiento de la red neuronal MLP
#
# IMPORTANTE:
# El conjunto TEST NO se utiliza para seleccionar
# hiperparámetros ni entrenar los modelos.
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

# Comprobar qué paquetes no están instalados
paquetes_faltantes <- paquetes[
  !paquetes %in% rownames(installed.packages())
]

# Instalar automáticamente los paquetes que falten
if (length(paquetes_faltantes) > 0) {
  
  cat("\nInstalando paquetes necesarios:\n")
  print(paquetes_faltantes)
  
  install.packages(
    paquetes_faltantes,
    dependencies = TRUE
  )
}

# Cargar librerías
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
# 3. CARGA DE LA PREPARACIÓN DEL SCRIPT 04
# ============================================================

source(
  file.path(
    ruta,
    "04_preparacion_datos_modelado.R"
  )
)

cat("\n\n")
cat("============================================\n")
cat("ENTRENAMIENTO DE LOS MODELOS PREDICTIVOS\n")
cat("============================================\n")


# ============================================================
# 4. 4.5.1 PREPARACIÓN DE LAS VARIABLES DE ENTRADA
# ============================================================

# Se utilizan exclusivamente los datos de TRAIN.
#
# X_train:
# variables originales para Random Forest y XGBoost.
#
# X_train_scaled:
# variables estandarizadas para la MLP.
#
# y_train:
# variable objetivo RUL.


datos_modelado <- X_train %>%
  mutate(
    RUL = y_train,
    unit_number = train$unit_number
  )


cat("\n--- PREPARACIÓN DE LAS VARIABLES DE ENTRADA ---\n")

cat(
  "Observaciones de entrenamiento:",
  nrow(X_train),
  "\n"
)

cat(
  "Variables predictoras:",
  ncol(X_train),
  "\n"
)

cat(
  "Motores disponibles:",
  n_distinct(train$unit_number),
  "\n"
)

cat("Variable objetivo: RUL\n")

cat(
  "Valores ausentes en X_train:",
  sum(is.na(X_train)),
  "\n"
)

cat(
  "Valores ausentes en y_train:",
  sum(is.na(y_train)),
  "\n"
)


# ============================================================
# 5. VALIDACIÓN CRUZADA AGRUPADA POR MOTOR
# ============================================================

# Se emplean 5 folds.
#
# La división se realiza por motor y no por observación.
# Todos los ciclos pertenecientes a un mismo motor quedan
# incluidos en el mismo fold.
#
# Esto evita que observaciones de un mismo motor aparezcan
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


datos_modelado <- datos_modelado %>%
  left_join(
    tabla_folds,
    by = "unit_number"
  )


cat("\n--- VALIDACIÓN CRUZADA ---\n")

cat("Número de folds: 5\n")
cat("Agrupación realizada por motor: SI\n")

cat("\nMotores por fold:\n")

print(
  tabla_folds %>%
    count(fold)
)


# ============================================================
# 6. FUNCIÓN PARA EL CÁLCULO DEL RMSE
# ============================================================

# Durante el ajuste de hiperparámetros se utiliza RMSE
# como criterio de selección dentro de TRAIN.

rmse <- function(real, predicho) {
  
  sqrt(
    mean(
      (real - predicho)^2
    )
  )
}


# ============================================================
# 7. 4.5.2 RANDOM FOREST
# ============================================================

cat("\n")
cat("============================================\n")
cat("RANDOM FOREST\n")
cat("============================================\n")


# ------------------------------------------------------------
# 7.1 Ajuste del hiperparámetro mtry
# ------------------------------------------------------------

# mtry representa el número de variables candidatas
# consideradas en cada división de los árboles.
#
# Se prueban cuatro configuraciones mediante validación
# cruzada agrupada por motor.

valores_mtry <- c(
  4,
  8,
  12,
  16
)


resultados_rf <- tibble()


for (mtry_actual in valores_mtry) {
  
  cat(
    "\nProbando Random Forest con mtry =",
    mtry_actual,
    "\n"
  )
  
  rmse_folds <- numeric(5)
  
  
  for (k in 1:5) {
    
    datos_train_fold <- datos_modelado %>%
      filter(fold != k)
    
    datos_valid_fold <- datos_modelado %>%
      filter(fold == k)
    
    
    modelo_fold <- ranger(
      
      RUL ~ .,
      
      data = datos_train_fold %>%
        select(
          all_of(predictores),
          RUL
        ),
      
      num.trees = 300,
      
      mtry = mtry_actual,
      
      min.node.size = 5,
      
      importance = "none",
      
      seed = 123
    )
    
    
    pred_fold <- predict(
      
      modelo_fold,
      
      data = datos_valid_fold %>%
        select(
          all_of(predictores)
        )
      
    )$predictions
    
    
    rmse_folds[k] <- rmse(
      datos_valid_fold$RUL,
      pred_fold
    )
    
    
    cat(
      "Fold",
      k,
      "- RMSE:",
      round(rmse_folds[k], 3),
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


cat("\n--- RESULTADOS DEL AJUSTE DE RANDOM FOREST ---\n")

print(resultados_rf)


# Selección del mejor mtry

mejor_mtry <- resultados_rf %>%
  slice_min(
    RMSE_CV,
    n = 1,
    with_ties = FALSE
  ) %>%
  pull(mtry)


cat(
  "\nMejor valor de mtry:",
  mejor_mtry,
  "\n"
)


# ------------------------------------------------------------
# 7.2 Entrenamiento del Random Forest definitivo
# ------------------------------------------------------------

# Una vez seleccionado mtry, el modelo definitivo se
# entrena utilizando todo TRAIN.

set.seed(123)

modelo_rf <- ranger(
  
  RUL ~ .,
  
  data = datos_modelado %>%
    select(
      all_of(predictores),
      RUL
    ),
  
  num.trees = 500,
  
  mtry = mejor_mtry,
  
  min.node.size = 5,
  
  importance = "permutation",
  
  seed = 123
)


cat("\nRandom Forest definitivo entrenado.\n")

cat(
  "Número de árboles:",
  modelo_rf$num.trees,
  "\n"
)

cat(
  "mtry:",
  modelo_rf$mtry,
  "\n"
)

cat(
  "min.node.size: 5\n"
)


# ------------------------------------------------------------
# 7.3 Importancia de variables Random Forest
# ------------------------------------------------------------

importancia_rf <- tibble(
  
  variable =
    names(modelo_rf$variable.importance),
  
  importancia =
    modelo_rf$variable.importance
  
) %>%
  arrange(
    desc(importancia)
  )


cat(
  "\n--- IMPORTANCIA DE VARIABLES: RANDOM FOREST ---\n"
)

print(
  importancia_rf,
  n = 24
)


# ============================================================
# 8. 4.5.3 XGBOOST
# ============================================================

cat("\n")
cat("============================================\n")
cat("XGBOOST\n")
cat("============================================\n")


# ------------------------------------------------------------
# 8.1 Conversión a matriz numérica
# ------------------------------------------------------------

X_train_xgb <- as.matrix(
  X_train
)

y_train_xgb <- y_train


# ------------------------------------------------------------
# 8.2 Configuraciones candidatas
# ------------------------------------------------------------

# Se ajustan dos hiperparámetros principales:
#
# max_depth:
# profundidad máxima de los árboles.
#
# eta:
# tasa de aprendizaje.
#
# Se mantiene:
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
# 8.3 Validación cruzada de XGBoost
# ------------------------------------------------------------

for (i in seq_len(nrow(config_xgb))) {
  
  cat(
    "\nProbando configuración XGBoost",
    config_xgb$configuracion[i],
    "\n"
  )
  
  cat(
    "max_depth =",
    config_xgb$max_depth[i],
    "- eta =",
    config_xgb$eta[i],
    "\n"
  )
  
  
  rmse_folds <- numeric(5)
  
  
  for (k in 1:5) {
    
    idx_train <- which(
      datos_modelado$fold != k
    )
    
    idx_valid <- which(
      datos_modelado$fold == k
    )
    
    
    dtrain <- xgb.DMatrix(
      
      data =
        X_train_xgb[idx_train, ],
      
      label =
        y_train_xgb[idx_train]
      
    )
    
    
    dvalid <- xgb.DMatrix(
      
      data =
        X_train_xgb[idx_valid, ],
      
      label =
        y_train_xgb[idx_valid]
      
    )
    
    
    parametros <- list(
      
      objective =
        "reg:squarederror",
      
      eval_metric =
        "rmse",
      
      max_depth =
        config_xgb$max_depth[i],
      
      eta =
        config_xgb$eta[i],
      
      subsample =
        0.8,
      
      colsample_bytree =
        0.8
      
    )
    
    
    modelo_fold <- xgb.train(
      
      params =
        parametros,
      
      data =
        dtrain,
      
      nrounds =
        300,
      
      verbose =
        0
      
    )
    
    
    pred_fold <- predict(
      modelo_fold,
      dvalid
    )
    
    
    rmse_folds[k] <- rmse(
      
      y_train_xgb[idx_valid],
      
      pred_fold
      
    )
    
    
    cat(
      "Fold",
      k,
      "- RMSE:",
      round(rmse_folds[k], 3),
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


cat(
  "\n--- RESULTADOS DEL AJUSTE DE XGBOOST ---\n"
)

print(
  resultados_xgb
)


# Selección de la mejor configuración

mejor_xgb <- resultados_xgb %>%
  slice_min(
    RMSE_CV,
    n = 1,
    with_ties = FALSE
  )


cat(
  "\nMejor configuración XGBoost:\n"
)

print(
  mejor_xgb
)


# ------------------------------------------------------------
# 8.4 Entrenamiento definitivo de XGBoost
# ------------------------------------------------------------

dtrain_completo <- xgb.DMatrix(
  
  data =
    X_train_xgb,
  
  label =
    y_train_xgb
  
)


parametros_finales_xgb <- list(
  
  objective =
    "reg:squarederror",
  
  eval_metric =
    "rmse",
  
  max_depth =
    mejor_xgb$max_depth,
  
  eta =
    mejor_xgb$eta,
  
  subsample =
    0.8,
  
  colsample_bytree =
    0.8
  
)


set.seed(123)


modelo_xgb <- xgb.train(
  
  params =
    parametros_finales_xgb,
  
  data =
    dtrain_completo,
  
  nrounds =
    300,
  
  verbose =
    0
  
)


cat(
  "\nXGBoost definitivo entrenado.\n"
)

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
  "Número de iteraciones: 300\n"
)


# ------------------------------------------------------------
# 8.5 Importancia de variables XGBoost
# ------------------------------------------------------------

importancia_xgb <- xgb.importance(
  model = modelo_xgb
)


cat(
  "\n--- IMPORTANCIA DE VARIABLES: XGBOOST ---\n"
)

print(
  importancia_xgb
)


# ============================================================
# 9. 4.5.4 RED NEURONAL MLP
# ============================================================

cat("\n")
cat("============================================\n")
cat("RED NEURONAL MLP\n")
cat("============================================\n")


# ------------------------------------------------------------
# 9.1 Datos estandarizados
# ------------------------------------------------------------

# La MLP utiliza la versión estandarizada de TRAIN
# preparada en el script 04.

X_train_mlp <- as.matrix(
  X_train_scaled
)

y_train_mlp <- y_train


# ------------------------------------------------------------
# 9.2 Configuraciones candidatas
# ------------------------------------------------------------

# Se utiliza una capa oculta.
#
# Se prueban tres tamaños diferentes:
#
# 5 neuronas
# 10 neuronas
# 15 neuronas

neuronas_candidatas <- c(
  5,
  10,
  15
)


resultados_mlp <- tibble()


# ------------------------------------------------------------
# 9.3 Validación cruzada de la MLP
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
      datos_modelado$fold != k
    )
    
    idx_valid <- which(
      datos_modelado$fold == k
    )
    
    
    set.seed(123)
    
    
    modelo_fold <- nnet(
      
      x =
        X_train_mlp[idx_train, ],
      
      y =
        y_train_mlp[idx_train],
      
      size =
        neuronas,
      
      linout =
        TRUE,
      
      decay =
        0.001,
      
      maxit =
        300,
      
      trace =
        FALSE,
      
      MaxNWts =
        10000
      
    )
    
    
    pred_fold <- predict(
      
      modelo_fold,
      
      X_train_mlp[idx_valid, ]
      
    )
    
    
    rmse_folds[k] <- rmse(
      
      y_train_mlp[idx_valid],
      
      pred_fold
      
    )
    
    
    cat(
      "Fold",
      k,
      "- RMSE:",
      round(rmse_folds[k], 3),
      "\n"
    )
  }
  
  
  resultados_mlp <- bind_rows(
    
    resultados_mlp,
    
    tibble(
      
      neuronas =
        neuronas,
      
      RMSE_CV =
        mean(rmse_folds),
      
      SD_RMSE =
        sd(rmse_folds)
      
    )
  )
}


cat(
  "\n--- RESULTADOS DEL AJUSTE DE LA MLP ---\n"
)

print(
  resultados_mlp
)


# Selección del número de neuronas

mejores_neuronas <- resultados_mlp %>%
  slice_min(
    RMSE_CV,
    n = 1,
    with_ties = FALSE
  ) %>%
  pull(neuronas)


cat(
  "\nNúmero de neuronas seleccionado:",
  mejores_neuronas,
  "\n"
)


# ------------------------------------------------------------
# 9.4 Entrenamiento definitivo de la MLP
# ------------------------------------------------------------

set.seed(123)


modelo_mlp <- nnet(
  
  x =
    X_train_mlp,
  
  y =
    y_train_mlp,
  
  size =
    mejores_neuronas,
  
  linout =
    TRUE,
  
  decay =
    0.001,
  
  maxit =
    500,
  
  trace =
    FALSE,
  
  MaxNWts =
    10000
  
)


cat(
  "\nMLP definitiva entrenada.\n"
)

cat(
  "Neuronas en la capa oculta:",
  mejores_neuronas,
  "\n"
)

cat(
  "Regularización (decay): 0.001\n"
)

cat(
  "Máximo de iteraciones: 500\n"
)


# ============================================================
# 10. RESUMEN FINAL DEL ENTRENAMIENTO
# ============================================================

cat("\n")
cat("============================================\n")
cat("RESUMEN DEL ENTRENAMIENTO\n")
cat("============================================\n")


cat("\nRANDOM FOREST\n")

cat(
  "mtry seleccionado:",
  mejor_mtry,
  "\n"
)

cat(
  "Número de árboles del modelo final: 500\n"
)

cat(
  "min.node.size: 5\n"
)


cat("\nXGBOOST\n")

cat(
  "max_depth seleccionado:",
  mejor_xgb$max_depth,
  "\n"
)

cat(
  "eta seleccionado:",
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
  "Número de iteraciones: 300\n"
)


cat("\nMLP\n")

cat(
  "Neuronas seleccionadas:",
  mejores_neuronas,
  "\n"
)

cat(
  "Número de capas ocultas: 1\n"
)

cat(
  "Regularización (decay): 0.001\n"
)

cat(
  "Máximo de iteraciones del modelo final: 500\n"
)


cat("\nVALIDACIÓN\n")

cat(
  "Validación cruzada: 5 folds\n"
)

cat(
  "Agrupación por motor: SI\n"
)

cat(
  "Criterio de selección: RMSE\n"
)

cat(
  "TEST utilizado durante entrenamiento o ajuste: NO\n"
)


cat("\n============================================\n")
cat("ENTRENAMIENTO FINALIZADO\n")
cat("============================================\n")
