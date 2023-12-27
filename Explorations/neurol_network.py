# Exercice :
#  reconstruire l'apprentissage d'un réseau de neuronne à propagation arrière
# qui comprend 2 couches d'entrées, 3 couches cachées et un neurone de sortie
# La fonction d'activation est une sigmoïde pour tous les neurones
# La fonction de coût est l'erreur quadratique
# L'optimisation se fait par une rétro-propagation du gradient d'erreur


import numpy as np
import matplotlib.pyplot as plt

# Definition des exemples entrees / sorties
# Couples de valeurs en entree :
# [0,0], [0.1,0], [0.2,0], ... [0.9,0], [1,0], [0,0.1], [0.1,0.1], ...
E1 = np.repeat(np.linspace(0,1,10).reshape([1,10]), 10, axis=0)
E1 = E1.reshape(100)
E2 = np.repeat(np.linspace(0,1,10),10)
E = np.array([E1, E2])

#  definition de la sortie Y que l'on souhaite
# 0 < E1 < 0,9 et 0 < E2 < 0,9
# Définition de la sortie Y que l'on souhaite
Y = np.zeros_like(E1)
# Condition pour Y = 1
condition = (E1 >= 0.5) & (E2 < 0.5) | (E1 < 0.5) & (E2 >= 0.5)
# Affecter 1 là où la condition est vraie
Y[condition] = 1

# Proposition CHELL
# for n_exe in range(0, E.shape[1]):
#    if (E[0, n_exe] >= 0.5 > E[1,n_exe]) or (E[0, n_exe] < 0.5 <= E[1,n_exe]):
#        Y[n_exe] = 1

# Il faut faire en sorte que la table de vérité soit la suivante :
#   in E1  in E2  / out Y
#     0       0       0
#     0.1     0       0
#     0.2     0       0
#     0.3     0       0
#     0.4     0       0
#     0.5     0       1
#     0.6     0       1
#     0.7     0       1
#     0.8     0       1
#     0.9     0       1
#     0       0.1       0
#     0.1     0.1       0
#     0.2     0.1       0
#     0.3     0.1       0
#     0.4     0.1       0
#     0.5     0.1       1
#     0.6     0.1       1
#     0.7     0.1       1
#     0.8     0.1       1
#     0.9     0.1       1
#   ...

# Affichage pour vérifier
print("E1     E2     Y")
for i in range(len(E1)):
    print(f"{E1[i]:.1f}    {E2[i]:.1f}    {Y[i]}")

plt.figure()
plt.plot(E[0, :])
plt.plot(E[1, :])
plt.plot(Y)
plt.legend(('E1', 'E2', 'Y'))
plt.show()

# Definition du reseau de neurones
V = np.random.rand(3, 3) # 3 (2+1) valeurs en entree et 3 neurones caches
W = np.random.rand(1, 4) # 4 (3+1) valeurs en entree de la couche de sortie

# Algorithme d’entrainement
N_iteration = 1000 # Nombre d’iterations de l’algorithme d'entrainement
N_exemples = E.shape[1] # Nombre d’exemples d’entrainement
alpha = 0.5 # taux apprentissage de l'algorithme
Err_quad_moyenne = np.zeros(N_iteration) # Vecteur pour enregistrer l'erreur à chaque iteration
for n_ite in range(0, N_iteration):
    for n_exe in range(0, N_exemples):
        #calculer la sortie de la premiere couche
        input = np.concatenate(([1], E[:, n_exe])) # ajout d'une entree fictive pour le biais
        #calculer la sortie de la couche 2
        Sv = 1/(1 + np.exp(-np.dot(V, input)))
        Sv = np.concatenate(([1], Sv))
        Sw = 1/(1 + np.exp(-np.dot(W, Sv)))

        plt.plot(Sv)
        plt.plot(Sw)
        #calculer l'erreur comise par le réseau
        Err_quad_moyenne[n_ite] = Err_quad_moyenne[n_ite] + 4#(Y[n_exe]-Sw)**2
        # calculer les correcteurs
        delta_W = Sw * (1 - Sw) * (Y[n_exe] - Sw)
        print("deltaW T shape", delta_W.T.shape)
        delta_V = Sv * (1 - Sv) * np.dot(delta_W.reshape((-1, 1)), W)
        print("deltaV T shape", delta_V.T.shape)
        print("Input T shape", E.T.shape)
        # mise à jour des poids du réseau
        V = V + alpha * np.dot(delta_V.T, E.T)
        W = W + alpha * np.dot(delta_W, Sv)
        # Enregistrement de l'erreur quadratique moyenne a cette iteration
        Err_quad_moyenne[n_ite] = Err_quad_moyenne[n_ite]/N_exemples

# affichage des resultats
x = range(0, n_ite)

# ...
plt.grid()
plt.title('Evolution de l''erreur quadratique moyenne')
plt.xlabel('Nombre d''iterations')
plt.ylabel('Erreur')
# Evolution des poids de la couche cachee
# ...
plt.grid()
plt.title('Evolution des poids de la couche cachee')

# Evolution des poids de la couche de sortie
plt.figure(3)
# ...
plt.grid()
plt.title('Evolution des poids de la couche de sortie')
plt.xlabel('Nombre d iterations')
plt.ylabel('Valeurs')

# Cartographie de la sortie
prediction = np.zeros([10,10])
attendue = np.zeros([10,10])
# for ii in range(0,10):
#     for kk in range(0, 10):
        # S1 = ...
        # S1 = ...
        # prediction[ii,kk] = ...
        # if (E[0, ii]>=0.5 and E[0,kk]<0.5) or (E[0, ii]<0.5 and E[0,kk]>=0.5):
        #     attendue[ii,kk] = 1

plt.figure(4)
plt.subplot(1,2,1)
plt.imshow(attendue,extent=[0, 1, 0, 1])
plt.title('Cartopgrahie de la fonction attendue')
plt.xlabel('Entree 1')
plt.ylabel('Entree 2')
plt.subplot(1,2,2)
plt.imshow(prediction,extent=[0, 1, 0, 1])
plt.title('Cartopgrahie de la fonction realisee')
plt.xlabel('Entree 1')
plt.ylabel('Entree 2')