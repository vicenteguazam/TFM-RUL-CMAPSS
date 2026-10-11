
# ============================================================
# TFM - PREDICCIÓN DE RUL EN MOTORES TURBOFAN
# 05b_entrenamiento_RF.R
#
# ENTRENAMIENTO DEL MODELO RANDOM FOREST
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
  "ranger"
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
cat("ENTRENAMIENTO DEL MODELO RANDOM FOREST\n")
cat("============================================\n")


# ============================================================
# 4. PREPARACIÓN DE LOS DATOS
# ============================================================

# Random Forest utiliza las variables predictoras
# en su escala original.

datos_rf <- X_train %>%
  mutate(
    RUL = y_train
  )


cat("\n--- DATOS DE ENTRENAMIENTO ---\n")

cat(
  "Observaciones:",
  nrow(datos_rf),
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
  "Valores ausentes:",
  sum(is.na(datos_rf)),
  "\n"
)


# ============================================================
# 5. CONFIGURACIÓN DEL MODELO
# ============================================================

mtry_rf <- 4
num_trees_rf <- 500
min_node_rf <- 5


cat("\n--- CONFIGURACIÓN RANDOM FOREST ---\n")

cat(
  "mtry:",
  mtry_rf,
  "\n"
)

cat(
  "Número de árboles:",
  num_trees_rf,
  "\n"
)

cat(
  "min.node.size:",
  min_node_rf,
  "\n"
)


# ============================================================
# 6. ENTRENAMIENTO DEL MODELO
# ============================================================

set.seed(123)

tiempo_inicio <- Sys.time()


modelo_rf <- ranger(
  
  RUL ~ .,
  
  data = datos_rf,
  
  num.trees = num_trees_rf,
  
  mtry = mtry_rf,
  
  min.node.size = min_node_rf,
  
  importance = "permutation",
  
  seed = 123
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
# 7. INFORMACIÓN DEL MODELO ENTRENADO
# ============================================================

cat("\n--- INFORMACIÓN DEL MODELO ---\n")

cat(
  "Número de árboles:",
  modelo_rf$num.trees,
  "\n"
)

cat(
  "Número de variables predictoras:",
  modelo_rf$num.independent.variables,
  "\n"
)

cat(
  "mtry:",
  modelo_rf$mtry,
  "\n"
)

cat(
  "Tamaño mínimo de nodo:",
  min_node_rf,
  "\n"
)


# ============================================================
# 8. IMPORTANCIA DE LAS VARIABLES
# ============================================================

importancia_rf <- tibble(
  
  variable =
    names(modelo_rf$variable.importance),
  
  importancia =
    as.numeric(modelo_rf$variable.importance)
  
) %>%
  
  arrange(
    desc(importancia)
  )


cat("\n--- IMPORTANCIA DE VARIABLES ---\n")

print(
  importancia_rf,
  n = nrow(importancia_rf)
)


cat("\n--- 10 VARIABLES MÁS IMPORTANTES ---\n")

print(
  importancia_rf %>%
    slice_head(n = 10)
)


# ============================================================
# 9. GRÁFICO DE IMPORTANCIA
# ============================================================

grafico_importancia_rf <- importancia_rf %>%
  
  slice_head(n = 10) %>%
  
  mutate(
    variable = reorder(
      variable,
      importancia
    )
  ) %>%
  
  ggplot(
    aes(
      x = importancia,
      y = variable
    )
  ) +
  
  geom_col() +
  
  labs(
    title = "Importancia de las variables en Random Forest",
    x = "Importancia por permutación",
    y = "Variable"
  ) +
  
  theme_minimal()


print(grafico_importancia_rf)


ggsave(
  
  filename = file.path(
    ruta,
    "importancia_variables_RF.png"
  ),
  
  plot = grafico_importancia_rf,
  
  width = 8,
  
  height = 5,
  
  dpi = 300
)


# ============================================================
# 10. GUARDADO DE RESULTADOS
# ============================================================

saveRDS(
  
  modelo_rf,
  
  file.path(
    ruta,
    "modelo_RF_final.rds"
  )
)


write.csv(
  
  importancia_rf,
  
  file.path(
    ruta,
    "importancia_variables_RF.csv"
  ),
  
  row.names = FALSE
)


cat("\n--- ARCHIVOS GENERADOS ---\n")

cat("modelo_RF_final.rds\n")
cat("importancia_variables_RF.csv\n")
cat("importancia_variables_RF.png\n")


# ============================================================
# 11. RESUMEN FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("RESUMEN DEL MODELO RANDOM FOREST\n")
cat("============================================\n")

cat(
  "Observaciones utilizadas:",
  nrow(datos_rf),
  "\n"
)

cat(
  "Predictores:",
  ncol(X_train),
  "\n"
)

cat(
  "mtry:",
  mtry_rf,
  "\n"
)

cat(
  "Número de árboles:",
  num_trees_rf,
  "\n"
)

cat(
  "min.node.size:",
  min_node_rf,
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
  importancia_rf %>%
    slice_head(n = 1)
)

cat("\n")
cat("============================================\n")
cat("ENTRENAMIENTO RANDOM FOREST FINALIZADO\n")
cat("============================================\n")
