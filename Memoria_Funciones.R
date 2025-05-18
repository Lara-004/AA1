suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(HDclassif))
suppressPackageStartupMessages(library(MASS))
suppressPackageStartupMessages(library(gt))
suppressPackageStartupMessages(library(knitr))
suppressPackageStartupMessages(library(cluster))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(factoextra))
suppressPackageStartupMessages(library (NbClust))
suppressPackageStartupMessages(library (parameters))
suppressPackageStartupMessages(library (stats))
suppressPackageStartupMessages(library(summarytools))
suppressPackageStartupMessages(library(corrplot))
suppressPackageStartupMessages(library(rpart))
suppressPackageStartupMessages(library(rpart.plot))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(xgboost))
suppressPackageStartupMessages(library(naivebayes))

#Bagging

#### Definimos los niveles manualmente

# Creamos lista con los niveles correctos para todas las variables categóricas
niveles_fijos <- list(
  Gender = c("Female", "Male"),
  family_history_with_overweight = c("yes", "no"),
  FAVC = c("yes", "no"),
  CAEC = c("no", "Sometimes", "Frequently", "Always"),
  SMOKE = c("yes", "no"),
  SCC = c("yes", "no"),
  CALC = c("no", "Sometimes", "Frequently", "Always"),
  MTRANS = c("Public_Transportation", "Walking", "Bike", "Motorbike", "Automobile"),
  NObeyesdad = levels(train$NObeyesdad)
)

#Recorremos todas las columnas del dataset y si en niveles_fijos hay niveles definidos los aplica ahí

for (col in names(train)) {
  if (!is.null(niveles_fijos[[col]])) {
    train[[col]] <- factor(train[[col]], levels = niveles_fijos[[col]])
  }
}



#### Entrenamos el modelo bagging
#Creamos esta función que entrena un modelo con bagging usando n_trees arboles
bagging_manual <- function(data, target, n_trees = 500)
  
{
  
  modelos <- list()     # Lista para guardar los árboles entrenados
  oob_indices <- list() # Lista para guardar los índices OOB de cada árbol
  set.seed(123)         #Para que los resultados sean reproducibles
  
  #Repetimos este bucle n_trees veces
  for(i in 1:n_trees) {
    n <- nrow(data)
    
    #Creamos una muestra con reemplazo
    muestra_idx <- sample(1:n, size = n, replace = TRUE)
    muestra <- data[muestra_idx, ]
    
    #Identificamos las observaciones que no han sido usadas. Estos son los datos   OOB que se usaran despues como validacion interna
    oob_idx <- setdiff(1:n, unique(muestra_idx))
    
    # Ajustamos los niveles de las variables categóricas para asegurarnos de que no falte ningun valor
    for (col in names(muestra)) {
      if (!is.null(niveles_fijos[[col]])) {
        muestra[[col]] <- factor(muestra[[col]], levels = niveles_fijos[[col]])
      }
    }
    
    
    #Creamos la fórmula del modelo:
    formula <- as.formula(paste(target, "~ ."))
    
    #Entrenamos un árbol de decisión con rpart
    arbol <- rpart(formula, data = muestra, method = "class")
    
    #Guardamos el árbol y los indices OOB
    modelos[[i]] <- arbol
    oob_indices[[i]] <- oob_idx
    
  }
  # Devolvemos todos los árboles y sus OOB correspondientes
  return(list(modelos = modelos, oob = oob_indices))
  
}


#### Predicción con observaciones Out Of the Bag

#Creamos una función que predice la clase de cada observacion usando solo los arboles donde esa observacion fue OOb
prediccion_oob <- function(modelos, oob_indices, data) {
  
  #Crea una lista vacía para guardar los "votos" que recibe cada observación (de  los árboles que la dejaron fuera al entrenar)
  n <- nrow(data)
  votos <- vector("list", n) # Guardaremos los votos para cada observación
  
  #Recorremos cada arbol y su correspondiente conjunto OOB
  for (i in seq_along(modelos)) {
    arbol <- modelos[[i]]
    oob <- oob_indices[[i]]
    #Extraemos los datos OOb para este arbol
    if(length(oob) > 0) {
      datos_oob <- data[oob, ]
      
      # Ajustamos los niveles para que coincidan con los del entrenamiento
      for (col in names(datos_oob)) {
        if (!is.null(niveles_fijos[[col]])) {
          datos_oob[[col]] <- factor(datos_oob[[col]], levels = niveles_fijos[[col]])
        }
      }
      # Hacemos la prediccion para las observaciones OOb usando el arbol actual
      pred <- predict(arbol, datos_oob, type = 'class')
      
      
      # Guardamos los votos por cada observación
      for (j in seq_along(oob)) {
        idx <- oob[j]
        votos[[idx]] <- c(votos[[idx]], as.character(pred[j]))
      }
    }
  }
  # Para cada observación, hacemos votación mayoritaria
  pred_final <- sapply(votos, function(v) {
    if (length(v) == 0) return(NA)  # Si no recibió ningún voto, lo dejamos en NA
    names(sort(table(v), decreasing = TRUE))[1]  # Clase más votada
  })
  
  return(factor(pred_final, levels = niveles_fijos$NObeyesdad))
  
}

# Naive Bayes

# Esta funcion entrena el modelo Naive Bayes usando conteos y probabilidales con el suavizado de Laplace
#Definimos una funcion que recibe el conjunto de datos, y el target
entrenar_naive_bayes <- function(data, target_col) {
  
  #Extraemos los valores unicos de la variable objetivo
  clases <- unique(data[[target_col]])
  
  #Guardamos el total de filas del dataset
  total <- nrow(data)
  
  #Guardamos todas las probabilidades calculadas para cada clase
  modelo <- list()
  
  #Recorremos cada clase posible (en nuestro caso de obesidad):
  for (clase in clases) {
    
    #Filtramos solos las filas que pertenecen a esa clase:
    subconjunto <- data[data[[target_col]] == clase, ]
    
    #Probabilidad a prori de esa clase (numero de ocurrencias entre el total):
    prob_clase <- nrow(subconjunto) / total
    
    #Guardamos la probabilidades condicionales para cada atributo:
    probs_atributos <- list()
    
    #Recorremos todas las columnas excepto la clase que vamos a predecir:
    for (col in colnames(data)) {
      
      if (col != target_col) {
        
        #Creamos una tabla de frecuencias para esa columna solo con las filas de la clase actual:
        tabla <- table(subconjunto[[col]])
        
        #Aplicamos el suavizado de Laplace que suma 1 a cada frecuencua para evitar ceros
        probs <- (tabla + 1) / (sum(tabla) + length(unique(data[[col]])))
        
        #Guardamos las probabilidades condicionales de ese atributo para esta clase
        probs_atributos[[col]] <- probs
      }
    }
    
    #Guardamos en la lista del modelo la probabilidad a prioti y las condcionales para cada variable
    modelo[[as.character(clase)]] <- list(
      prob = prob_clase,
      atributos = probs_atributos
    )
  }
  
  modelo$target <- target_col
  return(modelo)
}

#### Predicción con Naive Bayes
# Función para predecir una fila con el modelo entrenado
#Predice la clase para una nueva observacion usando log-probabilidades
predecir_naive_bayes <- function(modelo, nueva_obs) {
  target <- modelo$target
  clases <- names(modelo)[names(modelo) != "target"]
  
  #Donde almacenamos el log-probabilidad total para cada clase
  log_probs <- c()
  
  #Empieza a recorrer clases y toma la log de la probabilidad a priori
  for (clase in clases) {
    log_prob <- log(modelo[[clase]]$prob)
    
    #Recorremos cada variable de la observacion y tomamos su valor
    for (col in names(nueva_obs)) {
      valor <- as.character(nueva_obs[[col]])
      
      #Busca la probabilidad condicional de ese valor dado la clase
      if (col %in% names(modelo[[clase]]$atributos)) {
        probs_col <- modelo[[clase]]$atributos[[col]]
        prob_valor <- probs_col[valor]
        
        #Si ese valor no existia en el entrenamiento aplicamos Laplace para asignarle una probabilidad pequeña
        if (is.na(prob_valor)) {
          prob_valor <- 1 / (sum(probs_col) + length(probs_col))  # Laplace para valor desconocido
        }
        #Acumula la log-probabilidad total
        log_prob <- log_prob + log(prob_valor)
      }
    }
    #Guardamos la log-probabilidad total para esa clase
    log_probs[clase] <- log_prob
  }
  #Devuelve la clase con mayor log-probabilidad
  clase_predicha <- names(which.max(log_probs))
  return(clase_predicha)
}

#### Evaluación del modelo




