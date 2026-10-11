
# ============================================================
# TFM - PREDICCIÓN DE RUL EN MOTORES TURBOFAN
# 05d_entrenamiento_MLP.R
#
# ENTRENAMIENTO DE LA RED NEURONAL MLP
#
# El modelo se entrena sobre la totalidad del conjunto TRAIN
# utilizando los hiperparámetros seleccionados previamente.
#
# Para la MLP se utilizan las variables predictoras
# estandarizadas y la variable objetivo RUL estandarizada.
#
# El conjunto TEST no se utiliza en esta etapa.
# ============================================================


# ============================================================
# 1. LIBRERÍAS
# ============================================================

paquetes <- c(
  "tidyverse",
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
cat("ENTRENAMIENTO DE LA RED NEURONAL MLP\n")
cat("============================================\n")


# ============================================================
# 4. PREPARACIÓN DE LOS DATOS
# ============================================================

# Las variables predictoras ya fueron estandarizadas
# en el script de preparación de datos.

X_train_mlp <- as.matrix(X_train_scaled)

# Variable objetivo original
y_train_mlp_original <- as.numeric(y_train)


# ============================================================
# 4.1. ESTANDARIZACIÓN DE LA VARIABLE OBJETIVO RUL
# ============================================================

media_rul <- mean(y_train_mlp_original)
sd_rul <- sd(y_train_mlp_original)

if (sd_rul == 0) {
  stop("La desviación estándar del RUL es cero.")
}

y_train_mlp <- (
  y_train_mlp_original - media_rul
) / sd_rul


cat("\n--- DATOS DE ENTRENAMIENTO ---\n")

cat(
  "Observaciones:",
  nrow(X_train_mlp),
  "\n"
)

cat(
  "Variables predictoras:",
  ncol(X_train_mlp),
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
  sum(is.na(X_train_mlp)),
  "\n"
)

cat(
  "Valores ausentes en RUL:",
  sum(is.na(y_train_mlp)),
  "\n"
)

cat("Variables predictoras estandarizadas: SI\n")
cat("Variable objetivo RUL estandarizada: SI\n")

cat(
  "Media original del RUL:",
  round(media_rul, 4),
  "\n"
)

cat(
  "Desviación estándar original del RUL:",
  round(sd_rul, 4),
  "\n"
)

cat(
  "Media del RUL estandarizado:",
  round(mean(y_train_mlp), 6),
  "\n"
)

cat(
  "Desviación estándar del RUL estandarizado:",
  round(sd(y_train_mlp), 6),
  "\n"
)


# ============================================================
# 5. CONFIGURACIÓN DE LA RED
# ============================================================

neuronas_mlp <- 15
decay_mlp <- 0.001
maxit_mlp <- 5000


cat("\n--- CONFIGURACIÓN MLP ---\n")

cat(
  "Variables de entrada:",
  ncol(X_train_mlp),
  "\n"
)

cat("Número de capas ocultas: 1\n")

cat(
  "Neuronas en la capa oculta:",
  neuronas_mlp,
  "\n"
)

cat("Neuronas de salida: 1\n")

cat(
  "decay:",
  decay_mlp,
  "\n"
)

cat(
  "maxit:",
  maxit_mlp,
  "\n"
)

cat("Salida lineal: SI\n")


# ============================================================
# 6. ENTRENAMIENTO DEL MODELO
# ============================================================

set.seed(123)

tiempo_inicio <- Sys.time()


modelo_mlp <- nnet(
  
  x = X_train_mlp,
  
  y = y_train_mlp,
  
  size = neuronas_mlp,
  
  decay = decay_mlp,
  
  maxit = maxit_mlp,
  
  linout = TRUE,
  
  trace = FALSE,
  
  MaxNWts = 10000
)


tiempo_fin <- Sys.time()

tiempo_entrenamiento <- as.numeric(
  difftime(
    tiempo_fin,
    tiempo_inicio,
    units = "secs"
  )
)


cat("\nModelo entrenado.\n")

cat(
  "Tiempo de entrenamiento:",
  round(tiempo_entrenamiento, 2),
  "segundos\n"
)


# ============================================================
# 7. INFORMACIÓN DEL MODELO ENTRENADO
# ============================================================

cat("\n--- INFORMACIÓN DEL MODELO ---\n")

cat(
  "Arquitectura:",
  ncol(X_train_mlp),
  "-",
  neuronas_mlp,
  "- 1\n"
)

cat(
  "Número de pesos del modelo:",
  length(modelo_mlp$wts),
  "\n"
)

cat(
  "Convergencia:",
  ifelse(
    modelo_mlp$convergence == 0,
    "SI",
    "NO"
  ),
  "\n"
)

cat(
  "Código de convergencia:",
  modelo_mlp$convergence,
  "\n"
)

cat(
  "Valor final de la función objetivo:",
  modelo_mlp$value,
  "\n"
)


if (modelo_mlp$convergence != 0) {
  
  warning(
    paste(
      "La red neuronal MLP no ha alcanzado",
      "el criterio de convergencia."
    )
  )
  
}


# ============================================================
# 8. ESTRUCTURA DE LA RED
# ============================================================

estructura_mlp <- tibble(
  
  capa = c(
    "Entrada",
    "Oculta",
    "Salida"
  ),
  
  neuronas = c(
    ncol(X_train_mlp),
    neuronas_mlp,
    1
  )
)


cat("\n--- ESTRUCTURA DE LA RED ---\n")

print(estructura_mlp)


# ============================================================
# 9. PARÁMETROS DE ESCALADO DEL RUL
# ============================================================

escalado_rul <- tibble(
  
  parametro = c(
    "media",
    "desviacion_estandar"
  ),
  
  valor = c(
    media_rul,
    sd_rul
  )
)


cat("\n--- PARÁMETROS DE ESCALADO DEL RUL ---\n")

print(escalado_rul)


# ============================================================
# 10. GUARDADO DEL MODELO Y RESULTADOS
# ============================================================

saveRDS(
  
  modelo_mlp,
  
  file.path(
    ruta,
    "modelo_MLP_final.rds"
  )
)


write.csv(
  
  estructura_mlp,
  
  file.path(
    ruta,
    "estructura_MLP.csv"
  ),
  
  row.names = FALSE
)


write.csv(
  
  escalado_rul,
  
  file.path(
    ruta,
    "escalado_RUL_MLP.csv"
  ),
  
  row.names = FALSE
)


cat("\n--- ARCHIVOS GENERADOS ---\n")

cat("modelo_MLP_final.rds\n")
cat("estructura_MLP.csv\n")
cat("escalado_RUL_MLP.csv\n")


# ============================================================
# 11. RESUMEN FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("RESUMEN DE LA RED NEURONAL MLP\n")
cat("============================================\n")

cat(
  "Observaciones utilizadas:",
  nrow(X_train_mlp),
  "\n"
)

cat(
  "Predictores:",
  ncol(X_train_mlp),
  "\n"
)

cat("Variables predictoras estandarizadas: SI\n")
cat("Variable objetivo RUL estandarizada: SI\n")

cat(
  "Arquitectura:",
  ncol(X_train_mlp),
  "-",
  neuronas_mlp,
  "- 1\n"
)

cat(
  "decay:",
  decay_mlp,
  "\n"
)

cat(
  "maxit:",
  maxit_mlp,
  "\n"
)

cat(
  "Número de pesos:",
  length(modelo_mlp$wts),
  "\n"
)

cat(
  "Convergencia:",
  ifelse(
    modelo_mlp$convergence == 0,
    "SI",
    "NO"
  ),
  "\n"
)

cat(
  "Código de convergencia:",
  modelo_mlp$convergence,
  "\n"
)

cat(
  "Valor final de la función objetivo:",
  modelo_mlp$value,
  "\n"
)

cat(
  "Tiempo de entrenamiento:",
  round(tiempo_entrenamiento, 2),
  "segundos\n"
)

cat(
  "Media utilizada para escalar RUL:",
  round(media_rul, 4),
  "\n"
)

cat(
  "Desviación estándar utilizada para escalar RUL:",
  round(sd_rul, 4),
  "\n"
)

cat("TEST utilizado durante el entrenamiento: NO\n")


cat("\n")
cat("============================================\n")
cat("ENTRENAMIENTO MLP FINALIZADO\n")
cat("============================================\n")
