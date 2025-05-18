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

# Knn

# Función k-NN con distancia euclides
mi_knn <- function(train, test, cl, k = 5) {
  # Verificamos que las dimensiones estan bien
  if(nrow(train) != length(cl)) {
    stop("El número de filas de train debe coincidir con la longitud de cl")
  }
  
  # Convertir a matrices para evitar problemas con data frames
  train <- as.matrix(train)
  test <- as.matrix(test)
  cl <- as.vector(cl)
  
  # Comprobamos que train y test tienen el mismo número de columnas
  if(ncol(train) != ncol(test)) {
    stop("train y test deben tener el mismo número de columnas")
  }
  
  # Calculamos las distancias euclideas
  distancias <- sqrt(outer(
    rowSums(train^2), 
    rowSums(test^2), 
    "+"
  ) - 2 * tcrossprod(train, test))
  
  # Buscamos lo k, vecinos más cercanos
  vecinos <- apply(distancias, 2, function(x) order(x)[1:k])
  
  # Despues de buscarlo predecimos las clases más frecuentes.
  predicciones <- apply(vecinos, 2, function(indices) {
    clases_vecinos <- cl[indices]
    names(which.max(table(clases_vecinos)))
  })
  
  return(predicciones)
}


# Analisis Discriminante Lineal
mi_lda <- function(X, y) {
  # Convertir a matriz si es necesario
  if (!is.matrix(X)) X <- as.matrix(X)
  if (is.factor(y)) y <- as.character(y)
  
  classes <- unique(y)
  n_classes <- length(classes)
  n_features <- ncol(X)
  
  # 1. Calcular probabilidades a priori
  priors <- table(y)/length(y)
  
  # 2. Calcular medias por clase
  means <- matrix(0, nrow = n_classes, ncol = n_features)
  for (i in 1:n_classes) {
    means[i,] <- colMeans(X[y == classes[i],, drop = FALSE])
  }
  rownames(means) <- classes
  
  # 3. Calcular matriz de covarianza común
  cov_matrix <- matrix(0, nrow = n_features, ncol = n_features)
  for (i in 1:n_classes) {
    X_centered <- scale(X[y == classes[i],], center = means[i,], scale = FALSE)
    cov_matrix <- cov_matrix + crossprod(X_centered)
  }
  cov_matrix <- cov_matrix/(nrow(X) - n_classes)
  cov_inv <- solve(cov_matrix)
  
  # Estructura del modelo
  model <- list(
    classes = classes,
    priors = priors,
    means = means,
    cov_inv = cov_inv,
    n_features = n_features
  )
  class(model) <- "mi_lda_model"
  return(model)
}

predict.mi_lda_model <- function(object, newdata, ...) {
  if (!is.matrix(newdata)) newdata <- as.matrix(newdata)
  
  # Asegurar que newdata tenga las mismas características
  if (ncol(newdata) != object$n_features) {
    stop("Número de características no coincide")
  }
  
  # Calcular términos de la función discriminante
  n <- nrow(newdata)
  n_classes <- length(object$classes)
  discriminants <- matrix(0, nrow = n, ncol = n_classes)
  
  for (i in 1:n_classes) {
    # Término 1: x^T Σ^{-1} μ_c
    term1 <- newdata %*% object$cov_inv %*% object$means[i,]
    
    # Término 2: (1/2) μ_c^T Σ^{-1} μ_c (escalar para cada clase)
    term2 <- 0.5 * drop(object$means[i,] %*% object$cov_inv %*% object$means[i,])
    
    # Término 3: log(P(Y=c)) (escalar para cada clase)
    term3 <- log(object$priors[i])
    
    # Combinar términos (R hace el broadcasting correctamente)
    discriminants[,i] <- term1 - term2 + term3
  }
  
  # Predecir la clase con mayor discriminante
  object$classes[max.col(discriminants)]
}


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

#### Boosting 

copy_train <- train
copy_test <- test
copy_train$y <- as.numeric(as.factor(copy_train$NObeyesdad)) - 1
copy_test$y <- as.numeric(factor(copy_test$NObeyesdad, levels = levels(as.factor(copy_train$NObeyesdad)))) - 1

# Número de clases
levels_lbl <- levels(as.factor(copy_train$NObeyesdad))
K <- length(levels_lbl)

# Pasar de factores a numéricos
tabla_preds <- setdiff(names(copy_train), c("NObeyesdad", "y"))
copy_train[tabla_preds] <- lapply(copy_train[tabla_preds], function(z) if(is.numeric(z)) z else as.numeric(as.factor(z)))
copy_test[tabla_preds]<- lapply(copy_test[tabla_preds],  function(z) if(is.numeric(z)) z else as.numeric(as.factor(z)))

df_train <- copy_train[c(tabla_preds, "y")]
df_train$y <- factor(df_train$y, levels = 0:(K-1))
df_test <- copy_test[tabla_preds]

# Función AdaBoost
adaboost_train_rpart <- function(df, M) {
  n <- nrow(df)                 # Numero de observaciones
  w <- rep(1/n, n)              # Pesos iniciales iguales para todos
  stumps <- vector("list", M)   # Lista para almacenar los árboles
  alphas <- numeric(M)          # Vector para los coeficientes alpha
  K <- length(levels(df$y))     # Número de clases
  
  # Entrenamiento de un árbol de profundidad 1
  for (m in seq_len(M)) {
    stump <- rpart(y ~ ., data = df, weights = w, control = rpart.control(maxdepth = 1, cp = 0, minsplit = 2))
    
    # Predicciones en el conjunto de train
    pred_factor <- predict(stump, df, type = "class")
    pred_num <- as.numeric(pred_factor) - 1
    y_num <- as.numeric(df$y) - 1
    
    # Error ponderado
    err <- sum(w * (pred_num != y_num))
    err <- pmin(pmax(err, 1e-10), 1 - 1e-10)
    
    # Coeficiente alpha ajustado para multiclase
    alphas[m] <- log((1 - err)/err) + log(K - 1)
    
    # Actualizar pesos
    w <- w * exp(alphas[m] * (pred_num != y_num))
    w <- w / sum(w)
    
    # Guardar el árbol entrenado
    stumps[[m]] <- stump
  }
  list(stumps = stumps, alphas = alphas, K = K)
}

# Función para predecir con el modelo AdaBoost entrenado
adaboost_predict_rpart <- function(model, df) {
  M <- length(model$alphas)        # Número de árboles entrenados
  n <- nrow(df)                    # Número de observaciones a predecir
  scores <- matrix(0, n, model$K)  # Matriz de puntuaciones por clase
  
  for (m in seq_len(M)) {
    # Predicción de cada árbol
    pred_factor <- predict(model$stumps[[m]], df, type = "class")
    pred_num <- as.numeric(pred_factor) - 1
    # Acumular la "contrubución" del arbol a la puntuación de cada clase
    for (k in 0:(model$K - 1)) {scores[, k+1] <- scores[, k+1] + model$alphas[m] * (pred_num == k)}
  }
  # La predicción es la clase con más puntuaciones
  apply(scores, 1, which.max) - 1
}

# Tiempo
time_taken <- system.time({model_ab <- adaboost_train_rpart(df_train, M = 10)})
# cat(sprintf("Tiempo de entrenamiento: %.2f s\n", time_taken[3]))

# Predicciones y accuracy
datos_pred <- df_test
preds <- adaboost_predict_rpart(model_ab, datos_pred)
accuracy <- mean(preds == copy_test$y) * 100
cat(sprintf("Accuracy en test: %.2f%%\n", accuracy))


