
# ============================================================
# TFM - PREDICCIÓN DE RUL EN MOTORES TURBOFAN
# 06_evaluacion_comparacion_modelos.R
#
# EVALUACIÓN Y COMPARACIÓN DE LOS MODELOS
#
# Modelos evaluados:
# - Random Forest
# - XGBoost
# - MLP
#
# Los modelos se aplican sobre las observaciones finales
# de los motores del conjunto TEST.
#
# Métricas:
# - MAE
# - RMSE
# - R2
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
# 3. PREPARACIÓN DE LOS DATOS
# ============================================================

source(
  file.path(
    ruta,
    "04_preparacion_datos_modelado.R"
  )
)


# ============================================================
# 4. CONVERSIÓN DEL RUL REAL DE TEST A VECTOR NUMÉRICO
# ============================================================

# rul_test se genera en el script de preparación como
# un objeto de una columna. Se extrae dicha columna para
# trabajar con un vector numérico de 259 valores.

if (is.data.frame(rul_test)) {
  
  rul_test_vector <- as.numeric(rul_test[[1]])
  
} else {
  
  rul_test_vector <- as.numeric(rul_test)
}


cat("\n\n")
cat("============================================\n")
cat("EVALUACIÓN Y COMPARACIÓN DE LOS MODELOS\n")
cat("============================================\n")


# ============================================================
# 5. CARGA DE LOS MODELOS ENTRENADOS
# ============================================================

modelo_RF <- readRDS(
  file.path(
    ruta,
    "modelo_RF_final.rds"
  )
)

modelo_XGBoost <- readRDS(
  file.path(
    ruta,
    "modelo_XGBoost_final.rds"
  )
)

modelo_MLP <- readRDS(
  file.path(
    ruta,
    "modelo_MLP_final.rds"
  )
)


cat("\n--- MODELOS CARGADOS ---\n")

cat("Random Forest: SI\n")
cat("XGBoost: SI\n")
cat("MLP: SI\n")


# ============================================================
# 6. COMPROBACIÓN DEL CONJUNTO TEST
# ============================================================

cat("\n--- CONJUNTO TEST ---\n")

cat(
  "Motores evaluados:",
  nrow(X_test),
  "\n"
)

cat(
  "Variables predictoras:",
  ncol(X_test),
  "\n"
)

cat(
  "Valores reales de RUL:",
  length(rul_test_vector),
  "\n"
)

cat(
  "RUL real mínima:",
  min(rul_test_vector),
  "\n"
)

cat(
  "RUL real máxima:",
  max(rul_test_vector),
  "\n"
)

cat(
  "RUL real media:",
  round(mean(rul_test_vector), 2),
  "\n"
)

cat(
  "Valores ausentes X_test:",
  sum(is.na(X_test)),
  "\n"
)

cat(
  "Valores ausentes X_test_scaled:",
  sum(is.na(X_test_scaled)),
  "\n"
)

cat(
  "Valores ausentes RUL TEST:",
  sum(is.na(rul_test_vector)),
  "\n"
)


# Comprobación de coherencia

if (
  nrow(X_test) != length(rul_test_vector)
) {
  
  stop(
    "El número de observaciones de X_test no coincide ",
    "con el número de valores reales de RUL."
  )
}


# ============================================================
# 7. PREDICCIÓN RANDOM FOREST
# ============================================================

cat("\n")
cat("============================================\n")
cat("PREDICCIÓN RANDOM FOREST\n")
cat("============================================\n")


pred_RF <- predict(
  modelo_RF,
  data = X_test
)$predictions


pred_RF <- as.numeric(pred_RF)


cat(
  "Predicciones generadas:",
  length(pred_RF),
  "\n"
)

cat(
  "RUL predicha mínima:",
  round(min(pred_RF), 2),
  "\n"
)

cat(
  "RUL predicha máxima:",
  round(max(pred_RF), 2),
  "\n"
)

cat(
  "RUL predicha media:",
  round(mean(pred_RF), 2),
  "\n"
)


# ============================================================
# 8. PREDICCIÓN XGBOOST
# ============================================================

cat("\n")
cat("============================================\n")
cat("PREDICCIÓN XGBOOST\n")
cat("============================================\n")


X_test_xgb <- as.matrix(X_test)


pred_XGBoost <- predict(
  modelo_XGBoost,
  X_test_xgb
)


pred_XGBoost <- as.numeric(pred_XGBoost)


cat(
  "Predicciones generadas:",
  length(pred_XGBoost),
  "\n"
)

cat(
  "RUL predicha mínima:",
  round(min(pred_XGBoost), 2),
  "\n"
)

cat(
  "RUL predicha máxima:",
  round(max(pred_XGBoost), 2),
  "\n"
)

cat(
  "RUL predicha media:",
  round(mean(pred_XGBoost), 2),
  "\n"
)


# ============================================================
# 9. PREDICCIÓN MLP
# ============================================================

cat("\n")
cat("============================================\n")
cat("PREDICCIÓN MLP\n")
cat("============================================\n")


X_test_mlp <- as.matrix(X_test_scaled)


# La MLP genera inicialmente la predicción
# en la escala estandarizada del RUL.

pred_MLP_scaled <- predict(
  modelo_MLP,
  X_test_mlp,
  type = "raw"
)


pred_MLP_scaled <- as.numeric(
  pred_MLP_scaled
)


# ============================================================
# 10. DESESTANDARIZACIÓN DE LAS PREDICCIONES MLP
# ============================================================

escalado_RUL <- read.csv(
  file.path(
    ruta,
    "escalado_RUL_MLP.csv"
  )
)


media_RUL <- escalado_RUL$valor[
  escalado_RUL$parametro == "media"
]

sd_RUL <- escalado_RUL$valor[
  escalado_RUL$parametro == "desviacion_estandar"
]


if (
  length(media_RUL) != 1 ||
  length(sd_RUL) != 1
) {
  
  stop(
    "No se han podido recuperar correctamente ",
    "los parámetros de escalado del RUL."
  )
}


pred_MLP <- (
  pred_MLP_scaled * sd_RUL
) + media_RUL


pred_MLP <- as.numeric(pred_MLP)


cat(
  "Predicciones generadas:",
  length(pred_MLP),
  "\n"
)

cat(
  "Media utilizada para desestandarizar:",
  round(media_RUL, 4),
  "\n"
)

cat(
  "Desviación estándar utilizada:",
  round(sd_RUL, 4),
  "\n"
)

cat(
  "RUL predicha mínima:",
  round(min(pred_MLP), 2),
  "\n"
)

cat(
  "RUL predicha máxima:",
  round(max(pred_MLP), 2),
  "\n"
)

cat(
  "RUL predicha media:",
  round(mean(pred_MLP), 2),
  "\n"
)


# ============================================================
# 11. COMPROBACIÓN DE LAS PREDICCIONES
# ============================================================

if (
  length(pred_RF) != length(rul_test_vector) ||
  length(pred_XGBoost) != length(rul_test_vector) ||
  length(pred_MLP) != length(rul_test_vector)
) {
  
  stop(
    "El número de predicciones de alguno de los modelos ",
    "no coincide con el número de valores reales de RUL."
  )
}


cat("\n--- COMPROBACIÓN DE PREDICCIONES ---\n")

cat(
  "RUL reales:",
  length(rul_test_vector),
  "\n"
)

cat(
  "Predicciones Random Forest:",
  length(pred_RF),
  "\n"
)

cat(
  "Predicciones XGBoost:",
  length(pred_XGBoost),
  "\n"
)

cat(
  "Predicciones MLP:",
  length(pred_MLP),
  "\n"
)

cat(
  "Dimensiones compatibles: SI\n"
)


# ============================================================
# 12. FUNCIÓN PARA EL CÁLCULO DE MÉTRICAS
# ============================================================

calcular_metricas <- function(
    real,
    predicho
) {
  
  MAE <- mean(
    abs(
      real - predicho
    )
  )
  
  
  RMSE <- sqrt(
    mean(
      (
        real - predicho
      )^2
    )
  )
  
  
  R2 <- 1 - (
    
    sum(
      (
        real - predicho
      )^2
    ) /
      
      sum(
        (
          real - mean(real)
        )^2
      )
    
  )
  
  
  return(
    c(
      MAE = MAE,
      RMSE = RMSE,
      R2 = R2
    )
  )
}


# ============================================================
# 13. CÁLCULO DE MÉTRICAS
# ============================================================

metricas_RF <- calcular_metricas(
  rul_test_vector,
  pred_RF
)


metricas_XGBoost <- calcular_metricas(
  rul_test_vector,
  pred_XGBoost
)


metricas_MLP <- calcular_metricas(
  rul_test_vector,
  pred_MLP
)


resultados_metricas <- tibble(
  
  modelo = c(
    "Random Forest",
    "XGBoost",
    "MLP"
  ),
  
  MAE = c(
    metricas_RF["MAE"],
    metricas_XGBoost["MAE"],
    metricas_MLP["MAE"]
  ),
  
  RMSE = c(
    metricas_RF["RMSE"],
    metricas_XGBoost["RMSE"],
    metricas_MLP["RMSE"]
  ),
  
  R2 = c(
    metricas_RF["R2"],
    metricas_XGBoost["R2"],
    metricas_MLP["R2"]
  )
)


cat("\n")
cat("============================================\n")
cat("RESULTADOS DE LA EVALUACIÓN\n")
cat("============================================\n")


print(
  
  resultados_metricas %>%
    
    mutate(
      MAE = round(MAE, 3),
      RMSE = round(RMSE, 3),
      R2 = round(R2, 4)
    )
)


# ============================================================
# 14. PREDICCIONES POR MOTOR
# ============================================================

resultados_predicciones <- tibble(
  
  motor = test_final$unit_number,
  
  RUL_real = rul_test_vector,
  
  RUL_RF = pred_RF,
  
  RUL_XGBoost = pred_XGBoost,
  
  RUL_MLP = pred_MLP
)


# ============================================================
# 15. CÁLCULO DE ERRORES POR MOTOR
# ============================================================

resultados_predicciones <- resultados_predicciones %>%
  
  mutate(
    
    error_RF =
      RUL_RF - RUL_real,
    
    error_XGBoost =
      RUL_XGBoost - RUL_real,
    
    error_MLP =
      RUL_MLP - RUL_real,
    
    
    error_abs_RF =
      abs(error_RF),
    
    error_abs_XGBoost =
      abs(error_XGBoost),
    
    error_abs_MLP =
      abs(error_MLP)
    
  )


cat("\n--- PRIMERAS PREDICCIONES ---\n")


print(
  
  resultados_predicciones %>%
    
    select(
      motor,
      RUL_real,
      RUL_RF,
      RUL_XGBoost,
      RUL_MLP
    ) %>%
    
    head(10)
)


# ============================================================
# 16. MODELO CON MENOR RMSE
# ============================================================

mejor_modelo <- resultados_metricas %>%
  
  arrange(RMSE) %>%
  
  slice(1)


cat("\n--- MODELO CON MENOR RMSE EN TEST ---\n")


print(
  
  mejor_modelo %>%
    
    mutate(
      MAE = round(MAE, 3),
      RMSE = round(RMSE, 3),
      R2 = round(R2, 4)
    )
)


# ============================================================
# 17. GRÁFICA RUL REAL FRENTE A RUL PREDICHA
# ============================================================

datos_grafica <- resultados_predicciones %>%
  
  select(
    motor,
    RUL_real,
    RUL_RF,
    RUL_XGBoost,
    RUL_MLP
  ) %>%
  
  pivot_longer(
    
    cols = c(
      RUL_RF,
      RUL_XGBoost,
      RUL_MLP
    ),
    
    names_to = "modelo",
    
    values_to = "RUL_predicha"
    
  )


grafica_comparacion <- ggplot(
  
  datos_grafica,
  
  aes(
    x = RUL_real,
    y = RUL_predicha
  )
  
) +
  
  geom_point(
    alpha = 0.55
  ) +
  
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +
  
  facet_wrap(
    ~ modelo
  ) +
  
  labs(
    title = "RUL real frente a RUL predicha",
    x = "RUL real (ciclos)",
    y = "RUL predicha (ciclos)"
  ) +
  
  theme_minimal()


ggsave(
  
  filename = file.path(
    ruta,
    "comparacion_RUL_real_predicha.png"
  ),
  
  plot = grafica_comparacion,
  
  width = 10,
  height = 5.5,
  dpi = 300
)


# ============================================================
# 18. GUARDADO DE RESULTADOS
# ============================================================

write.csv(
  
  resultados_metricas,
  
  file.path(
    ruta,
    "metricas_modelos_TEST.csv"
  ),
  
  row.names = FALSE
)


write.csv(
  
  resultados_predicciones,
  
  file.path(
    ruta,
    "predicciones_modelos_TEST.csv"
  ),
  
  row.names = FALSE
)


cat("\n--- ARCHIVOS GENERADOS ---\n")

cat("metricas_modelos_TEST.csv\n")
cat("predicciones_modelos_TEST.csv\n")
cat("comparacion_RUL_real_predicha.png\n")


# ============================================================
# 19. RESUMEN FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("RESUMEN DE LA EVALUACIÓN FINAL\n")
cat("============================================\n")


cat(
  "Motores evaluados:",
  length(rul_test_vector),
  "\n"
)


cat(
  "Modelos evaluados: 3\n"
)


cat(
  "Métricas utilizadas: MAE, RMSE y R2\n"
)


cat(
  "Modelo con menor RMSE:",
  mejor_modelo$modelo,
  "\n"
)


cat(
  "RMSE:",
  round(
    mejor_modelo$RMSE,
    3
  ),
  "ciclos\n"
)


cat(
  "MAE:",
  round(
    mejor_modelo$MAE,
    3
  ),
  "ciclos\n"
)


cat(
  "R2:",
  round(
    mejor_modelo$R2,
    4
  ),
  "\n"
)


cat("\n")
cat("============================================\n")
cat("EVALUACIÓN Y COMPARACIÓN FINALIZADA\n")
cat("============================================\n")
