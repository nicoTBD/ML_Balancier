# Projet Balancier Machine Learning
The aim of the project is to predict if a pendulum is mooving or not. We use keras, tensorflow machine learning plugins.

# **Développeurs** 

LEROY Corentin
TABARAUD Nicolas

# **Fonctionnement Projet Machine Learning**


## **1) Concatenation**

Le fichier concatenate_csv_files permet de concaténer les différents fichiers d'acquisition des données avec le capteur Sensor tile

Un nouveau dossier se crée dans lequel on retrouve le fichier csv avec 23000 données environ

## **2) Normalisation et labelisation**

Il faut alors traiter ces données avant la labélisation, on applique une normalisation des données entre -1 et 1, avant de labéliser les données.

Pour labeliser les données, nous avons décider de manière arbitraire un certain intervalle de confiance pour chacun des axes X, Y, Z des données d'accélération.

Egalement, il y a un pré-traitement des données :  on supprime les données au début et à la fin de notre fichier csv et on décalle les données pour qu'elles soient toutes à la suite.

```
confidence_intervals = {
        'X': (0.18, 0.25),
        'Y': (0.28, 0.36),
        'Z': (0.092, 0.126)
    }
```

Pour déterminer ces intervalles de confiance nous avons utiliser le programme graph_histogram qui nous a donné la répartition des points.

Par ailleurs, nous avions des erreurs d'acquisition au tout début du graphe des données et à la toute fin. Nous avons donc supprimé ces données qui nuisent à la labellisation. 
Mais aussi, nous avons réassigné les valeurs pour éviter les “trous” dans les données dues à un problème d'acquisition de la “sensor tile”. Nous avons donc décalé toutes les données pour qu'elles se suivent sur le graphe.
Enfin, nous avons labellisé les données en créant un script python qui parcourt toutes les données du fichier “donnees_concatenes.csv” par fenêtre de 72 données. Ce nombre, nous l’avons trouvé de manière empirique. Au départ nous l’avions fixé à 6 mais après analyse des graphes nous avons augmenté ce nombre de step pour arriver finalement à une fenêtre de 72 données.

La boucle for itère sur les données d'accélération avec un pas de step (72 dans ce cas).
Pour chaque itération, on extrait la fenêtre de données de la série temporelle, de la position actuelle jusqu'à i + step.
On compte le nombre de données qui dépassent les bornes de confiance spécifiées dans cette fenêtre (confidence_interval_high et confidence_interval_low).
On labellise à 1 si la majorité des données dépassent ces bornes, sinon on labellise à 0.
On ajoute l'étiquette assignée à la liste “labels” pour chaque élément dans la fenêtre (“step” fois).

### **N.B.**

Résumé du processus de labellisation : 
concaténation de tous les fichiers d'acquisition de données de la “sensor tile”
affichage des graphes des 3 axes X, Y, Z
normalisation des données entre -1 et 1
affichage des histogrammes pour déterminer les seuils de confiance
script de labellisation des données avec élection via majorité sur une fenêtre de parcours des données
affichage de la labellisation finale


## **3) Validation des données**

Nous avons créée un script pour visualiser nos données sous forme de FFT

## **4) Creation du modèle**

Après plusieurs essaies et compréhension de notre problème nous nous sommes fixés sur une fonction de perte de "binary_crossentropy" pour avoir un perceptron de sortie avec un résultat binaire (0 : statique ou 1 : mouvement) 

En outre, nous avions commencé avec un modèle 2-3-1 mais les résultats n'étaient pas du tout satisfaisant.

Nous avons essayé de jouer sur les fonctions d'activation internes et sommes restés sur une fonction "relu" avec laquelle nous obtenions les meilleurs résultats.

## **5) Ajustement**

Nous avons essayés de jouer sur le learning_rate afin qu'il puisse décroitre au fur et à mesure des époques.

Enfin en terme d'optimiseur, nous avions commencé par Adam combinant les avantages de l'optimiseur RMSprop et de l'optimiseur de descente de gradient stochastique (SGD). 
Mais nous avons obtenu des meilleurs résultats avec le SGD (Stochastic Gradient Descent) qui est le plus simple des optimiseurs mais il peut converger plus lentement que des optimiseurs plus avancés.

Nos résultats n'étaient pas à la hauteur de nos attentes donc nous avons ajouté plusieurs couches cachés où nous avons augmenté les perceptrons sur chaque couches internes. Mais aussi, nous avons ajouter des Dropout pour désactiver aléatoirement certains neuronnes, ce qui permet de stabiliser le modèle, et donc réduire l'overfitting.

Enfin, nous avons vu qu'en utilisant des fonctions de pénalisation l1 ou l2 pouvaient être utiles donc nous avons utilisé l2.
