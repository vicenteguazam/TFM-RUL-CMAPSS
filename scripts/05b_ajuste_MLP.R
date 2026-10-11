
# ============================================================
# TFM - PREDICCIÓN DE RUL EN MOTORES TURBOFAN
# AJUSTE DEL TAMAÑO DE LA RED NEURONAL MLP
#
# Objetivo:
# Evaluar diferentes números de neuronas en la capa oculta
# y seleccionar la configuración con menor RMSE mediante
# validación cruzada agrupada por motor.
# ============================================================


# ------------------------------------------------------------
# 1. LIBRERÍAS
# ------------------------------------------------------------

library(tidyverse)
library(nnet)


# ------------------------------------------------------------
# 2. RUTA DE TRABAJO
# ------------------------------------------------------------

if (file.exists("scripts/train_FD002_prepared.csv")) {
  setwd("scripts")
} else if (!file.exists("train_FD002_prepared.csv")) {
  stop("No se encuentra train_FD002_prepared.csv. Ejecuta desde la raíz del repositorio o desde scripts/.")
}

ruta <- "."


# ------------------------------------------------------------
# 3. CARGA Y PREPARACIÓN DE LOS DATOS
# ------------------------------------------------------------

source(
  file.path(
    ruta,
    "04_preparacion_datos_modelado.R"
  )
)

X_mlp <- as.matrix(X_train_scaled)
y_mlp <- y_train


cat("\n")
cat("============================================\n")
cat("AJUSTE DEL TAMAÑO DE LA RED NEURONAL MLP\n")
cat("============================================\n")

cat("\nObservaciones:", nrow(X_mlp), "\n")
cat("Variables predictoras:", ncol(X_mlp), "\n")
cat("Motores:", n_distinct(train$unit_number), "\n")


# ------------------------------------------------------------
# 4. VALIDACIÓN CRUZADA AGRUPADA POR MOTOR
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 5. FUNCIÓN RMSE
# ------------------------------------------------------------

rmse <- function(real, predicho) {
  
  sqrt(
    mean(
      (real - predicho)^2
    )
  )
}


# ------------------------------------------------------------
# 6. TAMAÑOS DE CAPA OCULTA EVALUADOS
# ------------------------------------------------------------

neuronas_candidatas <- c(
  15,
  20,
  25,
  30
)

resultados_mlp <- tibble()


# ------------------------------------------------------------
# 7. AJUSTE DEL NÚMERO DE NEURONAS
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
    
    
    modelo_mlp <- nnet(
      
      x = X_mlp[idx_train, ],
      
      y = y_mlp[idx_train],
      
      size = neuronas,
      
      linout = TRUE,
      
      decay = 0.001,
      
      maxit = 300,
      
      trace = FALSE,
      
      MaxNWts = 10000
    )
    
    
    predicciones <- predict(
      modelo_mlp,
      X_mlp[idx_valid, ]
    )
    
    
    rmse_folds[k] <- rmse(
      y_mlp[idx_valid],
      predicciones
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
      neuronas = neuronas,
      RMSE_CV = mean(rmse_folds),
      SD_RMSE = sd(rmse_folds)
    )
  )
}


# ------------------------------------------------------------
# 8. RESULTADOS
# ------------------------------------------------------------

cat("\n")
cat("============================================\n")
cat("RESULTADOS DEL AJUSTE DE LA MLP\n")
cat("============================================\n")

print(resultados_mlp)


# ------------------------------------------------------------
# 9. SELECCIÓN DEL TAMAÑO DE LA CAPA OCULTA
# ------------------------------------------------------------

mejor_configuracion <- resultados_mlp %>%
  slice_min(
    RMSE_CV,
    n = 1,
    with_ties = FALSE
  )


cat("\nConfiguración con menor RMSE:\n")

print(mejor_configuracion)


cat("\n")
cat("============================================\n")
cat("AJUSTE FINALIZADO\n")
cat("============================================\n")
