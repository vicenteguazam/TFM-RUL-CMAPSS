
# ============================================================
# TFM - Estimación de la vida útil remanente (RUL)
# Dataset C-MAPSS - FD002
#
# 3.4.1 Conversión y limpieza de datos
# ============================================================


# ------------------------------------------------------------
# 1. CARGA DE LIBRERÍAS
# ------------------------------------------------------------

library(tidyverse)


# ------------------------------------------------------------
# 2. IMPORTACIÓN DE LOS DATOS
# ------------------------------------------------------------

# Los archivos CSV utilizan ";" como separador.
# Se admiten ejecuciones desde la raíz del repositorio
# o directamente desde la carpeta scripts.

if (file.exists("scripts/train_FD002_prepared.csv")) {
  
  ruta_datos <- "scripts"
  
} else if (file.exists("train_FD002_prepared.csv")) {
  
  ruta_datos <- "."
  
} else {
  
  stop(
    "No se encuentra train_FD002_prepared.csv. ",
    "Ejecuta el script desde la raíz del repositorio ",
    "o desde la carpeta scripts."
  )
}

archivos_necesarios <- c(
  "train_FD002_prepared.csv",
  "test_FD002_prepared.csv"
)

for (archivo in archivos_necesarios) {
  
  if (!file.exists(file.path(ruta_datos, archivo))) {
    
    stop(
      "No se encuentra el archivo: ",
      file.path(ruta_datos, archivo)
    )
  }
}

train <- read_delim(
  file.path(ruta_datos, "train_FD002_prepared.csv"),
  delim = ";",
  locale = locale(decimal_mark = "."),
  show_col_types = FALSE
)

test <- read_delim(
  file.path(ruta_datos, "test_FD002_prepared.csv"),
  delim = ";",
  locale = locale(decimal_mark = "."),
  show_col_types = FALSE
)


# ------------------------------------------------------------
# 3. COMPROBACIÓN DE LAS DIMENSIONES
# ------------------------------------------------------------

cat("\n--- DIMENSIONES ---\n")

cat("TRAIN:\n")
print(dim(train))

cat("TEST:\n")
print(dim(test))


# ------------------------------------------------------------
# 4. COMPROBACIÓN DEL NÚMERO DE MOTORES
# ------------------------------------------------------------

cat("\n--- NÚMERO DE MOTORES ---\n")

cat("Motores en TRAIN:",
    n_distinct(train$unit_number), "\n")

cat("Motores en TEST:",
    n_distinct(test$unit_number), "\n")


# ------------------------------------------------------------
# 5. COMPROBACIÓN DE LOS NOMBRES DE LAS VARIABLES
# ------------------------------------------------------------

cat("\n--- VARIABLES TRAIN ---\n")
print(names(train))

cat("\n--- VARIABLES TEST ---\n")
print(names(test))


# ------------------------------------------------------------
# 6. COMPROBACIÓN DE LA ESTRUCTURA Y TIPOS DE DATOS
# ------------------------------------------------------------

cat("\n--- ESTRUCTURA TRAIN ---\n")
glimpse(train)

cat("\n--- ESTRUCTURA TEST ---\n")
glimpse(test)


# ------------------------------------------------------------
# 7. COMPROBACIÓN DE VALORES AUSENTES
# ------------------------------------------------------------

cat("\n--- VALORES AUSENTES ---\n")

cat("Valores ausentes totales en TRAIN:",
    sum(is.na(train)), "\n")

cat("Valores ausentes totales en TEST:",
    sum(is.na(test)), "\n")

cat("\nValores ausentes por variable en TRAIN:\n")
print(colSums(is.na(train)))

cat("\nValores ausentes por variable en TEST:\n")
print(colSums(is.na(test)))


# ------------------------------------------------------------
# 8. COMPROBACIÓN DE REGISTROS DUPLICADOS
# ------------------------------------------------------------

cat("\n--- REGISTROS DUPLICADOS ---\n")

cat("Registros duplicados en TRAIN:",
    sum(duplicated(train)), "\n")

cat("Registros duplicados en TEST:",
    sum(duplicated(test)), "\n")


# ------------------------------------------------------------
# 9. COMPROBACIÓN DE LA UNICIDAD MOTOR-CICLO
# ------------------------------------------------------------

duplicados_motor_ciclo_train <- train %>%
  count(unit_number, cycle) %>%
  filter(n > 1)

duplicados_motor_ciclo_test <- test %>%
  count(unit_number, cycle) %>%
  filter(n > 1)

cat("\n--- COMBINACIONES MOTOR-CICLO DUPLICADAS ---\n")

cat("TRAIN:",
    nrow(duplicados_motor_ciclo_train), "\n")

cat("TEST:",
    nrow(duplicados_motor_ciclo_test), "\n")


# ------------------------------------------------------------
# 10. RESUMEN DE LA COMPROBACIÓN
# ------------------------------------------------------------

cat("\n============================================\n")
cat("RESUMEN DE COMPROBACIÓN DE LOS DATOS\n")
cat("============================================\n")

cat("TRAIN:",
    nrow(train), "registros -",
    ncol(train), "variables -",
    n_distinct(train$unit_number), "motores\n")

cat("TEST:",
    nrow(test), "registros -",
    ncol(test), "variables -",
    n_distinct(test$unit_number), "motores\n")

cat("Valores ausentes TRAIN:", sum(is.na(train)), "\n")
cat("Valores ausentes TEST:", sum(is.na(test)), "\n")

cat("Duplicados TRAIN:", sum(duplicated(train)), "\n")
cat("Duplicados TEST:", sum(duplicated(test)), "\n")

cat("Duplicados motor-ciclo TRAIN:",
    nrow(duplicados_motor_ciclo_train), "\n")

cat("Duplicados motor-ciclo TEST:",
    nrow(duplicados_motor_ciclo_test), "\n")


# ============================================================
# 3.4.2 CONSTRUCCIÓN Y COMPROBACIÓN DE LA VARIABLE RUL
# ============================================================

# Cálculo del ciclo máximo alcanzado por cada motor y de la RUL

train_rul <- train %>%
  group_by(unit_number) %>%
  mutate(
    max_cycle = max(cycle),
    RUL_calculada = max_cycle - cycle
  ) %>%
  ungroup()


# ------------------------------------------------------------
# COMPROBACIÓN DE LA RUL
# ------------------------------------------------------------

# Comparación con la RUL previamente incorporada al dataset

diferencias_rul <- sum(train_rul$RUL != train_rul$RUL_calculada)

cat("\n--- COMPROBACIÓN DE LA RUL ---\n")
cat("Registros analizados:", nrow(train_rul), "\n")
cat("Motores analizados:", n_distinct(train_rul$unit_number), "\n")
cat("Diferencias entre RUL existente y RUL calculada:",
    diferencias_rul, "\n")


# ------------------------------------------------------------
# RANGO DE LA RUL
# ------------------------------------------------------------

cat("\n--- RANGO DE LA RUL ---\n")
cat("RUL mínima:", min(train_rul$RUL_calculada), "\n")
cat("RUL máxima:", max(train_rul$RUL_calculada), "\n")


# ------------------------------------------------------------
# COMPROBACIÓN DEL ÚLTIMO CICLO DE CADA MOTOR
# ------------------------------------------------------------

ultimo_ciclo <- train_rul %>%
  group_by(unit_number) %>%
  slice_max(cycle, n = 1, with_ties = FALSE) %>%
  ungroup()

cat("\n--- COMPROBACIÓN DEL ÚLTIMO CICLO ---\n")
cat("Motores con RUL final distinta de 0:",
    sum(ultimo_ciclo$RUL_calculada != 0), "\n")


# ------------------------------------------------------------
# EJEMPLO DE COMPROBACIÓN: MOTOR 1
# ------------------------------------------------------------

cat("\n--- EJEMPLO MOTOR 1 ---\n")

train_rul %>%
  filter(unit_number == 1) %>%
  select(unit_number, cycle, max_cycle, RUL, RUL_calculada) %>%
  slice_head(n = 10) %>%
  print()


# ------------------------------------------------------------
# RESUMEN
# ------------------------------------------------------------

cat("\n============================================\n")
cat("RESUMEN DE LA CONSTRUCCIÓN DE LA RUL\n")
cat("============================================\n")
cat("Registros:", nrow(train_rul), "\n")
cat("Motores:", n_distinct(train_rul$unit_number), "\n")
cat("RUL mínima:", min(train_rul$RUL_calculada), "\n")
cat("RUL máxima:", max(train_rul$RUL_calculada), "\n")
cat("Diferencias detectadas:", diferencias_rul, "\n")
cat("Motores con RUL final distinta de 0:",
    sum(ultimo_ciclo$RUL_calculada != 0), "\n")
