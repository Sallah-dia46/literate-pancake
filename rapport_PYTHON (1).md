RAPORT:  PACKAGING
I) INTRODUCTION
 Les packages (paquets) sont des modules qui contiennent d’autres modules d''où le nom PACKAGING.
 Un package correspond à un dossier dans un répertoire sur lequel on a créer d''autres fichiers python où on pouvons importer les modules créés.  
 Pour importer un package dans un programme, il faut que c package soit dans le même répertoire que le fichier qui contienne le programme. 
 Ainsi, nous allons exploiter plusieurs modules a savoir Numpy, Scipy, Matplotlib, Pandas,Seaborn etc 
II) NOTION DE MODULE
   Nous avons des modules universels qui sont en défaut des modules de python appelés aussi librairies(bibliothèque)
    et des modules personnels(création de son propre module et faire son importation dans un programme afin d''utiliser ses fonctions.)
   Un module est un fichier Python contenant des définitions et des instruction. Il peut contenir des classes avec des methodes ou des fonctions.
   prenez votre éditeur favori et créez un fichier fibo.py dans le répertoire courant où fibo est le nom du fichier indexé de '.py'
   importez le module et utilisez ces fonctions dans un autre fichier.py qui est dans ce même répertoire. 
 A) installation d''un module de python
    L''installation d''un module python se fait dans le terminal du répertoire utilisé
    ou de l'environnement utilisé avec la commande pip instal "suivi du nom de module ".
    exemple: pip instal SCIPY
 B)utulité(comment utiliser un module dans un répertoire et son sens)
    Après l'installation du module, on l'importe dans notre programme pour l'utiliser.
    exemple: import SCIPY as sp 
    après son importation, on l'utilise comme suit: "sp."suivi d'une de ces fonctions que l'on souhaite utiliser.
    On utilise Scipy pour la Data Science et d’autres domaines d’ingénierie,car il contient les fonctions optimisées nécessaires 
    pour résoudre une large variétés de problèmes scientifiques, chaque fonction de son bibliothèque open source a son utulité.
 C)EXEMPLES DE MODULES
  C_ 1)Numpy :c''est une bibliothèque de langage de programmation Python, destinée à manipuler des matrices ou tableaux multidimensionnels
    ainsi que des fonctions mathématiques opérant sur ces tableaux.
    La bibliothèque NumPy permet d’effectuer des calculs numériques avec Python. Elle introduit une gestion facilitée des tableaux de nombres.
    Elle contient plusieurs fonctions qui lui permettent d'offrir une large variétés de modèles de manipulation.
    importation: import numpy as np
    exemple de fonction numpy:
    # np.pi #  donne la valeur de pi
    a = np.array([1, 2, 3, 4]) 
     array qui signifie tableau en anglais, ce fonction nous permet de créer un tableau
     voici l'affichage obtenue: 
      ## array([1, 2, 3, 4])
      type(a)     
      numpy.ndarray
    b = np.array([[1, 2, 3], [4, 5, 6]]) 
    nous avons un tableau bidimentionels:
      ## array([[1, 2, 3],
       [4, 5, 6]])

    m = np.arange(3, 15, 2)
   # la fonction arange permet de ranger les nombres comme suit dans cette exemple
   # (de 3 à 15 n'affichant pas le 15 et respectant le saut de 2)   
   # voici le résultat:
      ##array([ 3,  5,  7,  9, 11, 13])
  # la fonction numpy.linspace() permet d’obtenir un tableau 1D
  # allant d’une valeur de départ à une valeur de fin avec un nombre donné d’éléments
   np.linspace(3, 9, 10)
   voivi le résultat:
   array([ 3.        ,  3.66666667,  4.33333333,  5.        ,  5.66666667,
        6.33333333,  7.        ,  7.66666667,  8.33333333,  9.        ])
  
  Numpy dispose d’un grand nombre de fonctions mathématiques qui peuvent être appliquées directement à un tableau.
  Dans ce cas, la fonction est appliquée à chacun des éléments du tableau
   x = np.linspace(-np.pi/2, np.pi/2, 3)
   x
    array([-1.57079633,  0.        ,  1.57079633])
   y = np.sin(x)
   y
   array([-1.,  0.,  1.])
 
  #la fonction np.around nous permet d'arrondir
     x = np.array([3.73637, 5.4374345]) # ici on a créé le tableau
     np.around(x,2)  #et là on arrondit à 2 chiffres apràs,la virgule
     array([ 3.74,  5.44])  #le résultat
  Nous allons maintenant à la découverte des choses plus sérieuses et merveilleuses avec numpy accompagné d''autres modules.
     
   C_2) matplotlib
  Matplotlib est une bibliothèque du langage de programmation Python destinée à tracer et visualiser des données sous formes de graphiques5. 
  Elle peut être combinée avec les bibliothèques python de calcul scientifique Numpy et SciPy. Elle fournit également une API orientée objet,
  permettant d''intégrer des graphiques dans des applications, utilisant des outils d''interface graphique polyvalents tels que Tkinter, wxPython, Qt
  Il regroupe un grand nombre de fonctions qui servent à créer des graphiques et les personnaliser (travailler sur les axes, le type de graphique, 
  sa forme et même rajouter du texte).
  l''installation de ce module peut se faire sous différentes systèmes a savoir lunix windows et Mag
        1 sous lunix 
 C’est sans doute sous Linux que matplotlib est le plus simple à installer. Il suffit d’utiliser
 son gestionnaire de paquets (en ligne de commande ou en graphique). Voici quelques exemples
 de commande d’installation.
        sudo pacman -S python-matplotlib # Sous Arch linux
2       sudo apt-get install python-matplotlib
.3. Installation sous Windows
Sous Windows, nous pouvons également utiliser pip pour installer matplotlib. Il nous suffit
donc d’ouvrir un terminal et d’entrer ces deux commandes. La première commande permet de
mettre à jour pip et la seconde installe matplotlib.
 py -m pip install --user -U --upgrade pip
 py -m pip install --user -U matplotlib
 apres avoir installer matplotlib nous nous interessons sur les traces 
 2.1. Ouvrir une fenêtre
Tout d’abord, importons le module pyplot. La plupart des gens ont l’habitude de l’importer
en tant que plt et nous ne dérogerons pas à la règle. On place donc cette ligne au début de
notre fichier.
1 import matplotlib.pyplot as plt
La première commande que nous allons voir dans ce module s’appelle show. Elle permet tout
simplement d’afficher un graphique dans une fenêtre. Par défaut, celui-ci est vide. Nous devrons
utiliser d’autres commandes pour définir ce que nous voulons afficher.
La seconde commande, close sert tout simplement à fermer la fenêtre qui s’est ouverte avec show.
Lorsque nous appuyons sur la croix de notre fenêtre, celle-ci se ferme également. Néanmoins, il
vaut mieux toujours utiliser close.
Finalement, voici notre premier code.
import matplotlib.pyplot as plt
3 plt.show()
4 plt.close()
En fait, la commande show sert bien à ouvrir la fenêtre, à «show» donc montrer ce que l’on
a fait précédemment. Mais nous n’avons rien fait. Nous devons alors introduire une troisième
commande, la commande plot. Finalement, voici notre vrai premier code.
import matplotlib.pyplot as plt
2
3 plt.plot()
4 plt.show()
5 plt.close()
Voilà, notre fenêtre s’ouvre bien devant nos yeux émerveillés. Nous pouvons regarder les options
offertes par la fenêtre dans le menu horizontal (zoom, déplacement, enregistrement en tant
qu’image…)

2.3. Tracer une figure
Cependant, nous pouvons aussi passer deux listes en arguments à plot. La première liste
correspondra à la liste des abscisses des points que nous voulons relier et la seconde à la liste de
leurs ordonnées. Ainsi, notre code précédent pourrait être le suivant.
import matplotlib.pyplot as plt
3 x = [0, 1, 2]
4 y = [1, 0, 2]
5 plt.plot(x, y)
6 plt.show()
7 plt.close()
Nous pouvons également passer en paramètre à plot plusieurs listes pour avoir plusieurs tracés. Par exemple avec ce code…
import matplotlib.pyplot as plt

x = [0, 1, 0]
y = [0, 1, 2]

x1 = [0, 2, 0]
y1 = [2, 1, 0]

x2 = [0, 1, 2]
y2 = [0, 1, 2]

plt.plot(x, y, x1, y1, x2, y2)
plt.show()
plt.close()

Les histogrammes
Pour créer un histogramme on utilise la méthode hist . On peut lui donner des données brutes et il
s'occupera de faire les calcules nécessaires à la présentation du graphique.
On a prit ici l'exemple de 1000 tirage au sort (random) de valeur entre 0 et 150 et voici le résultat:
import matplotlib.pyplot as plt
import random
# 1000 tirages entre 0 et 150
x = [random.randint(0,150) for i in range(1000)]
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='b', alpha=0.5)

plt.xlabel('Mise')
plt.ylabel(u'Probabilité')
plt.axis([0, 150, 0, 0.02])
plt.grid(True)
plt.show()


C_3) pandas
   Pandas est une bibliothèque écrite pour le langage de programmation Python permettant la manipulation et l'analyse des données. 
Elle propose en particulier des structures de données et des opérations de manipulation de tableaux numériques et de séries temporelles.
 pour installer pandas , il faudra verifier l'existence de python dans l'ordinateur, ainsi, on utilisera la commande pip pour l'installer 

pip install pandas

ou bien aller directement telecharger Anaconda Navigator

     ######     Creation des graphiques #####

Pandas nous permet de creer nos dataframes.Il va vous permettre de charger votre dataset et de manipuler toutes nos données comme nous le voulons.
Ainsi, nous allons faire une presentation de  quelques grahiques en prenant comme exemple l'age de trois personnes

  ## importation de la library pandas ##
import pandas as pd
 ## importation de la library matplotlib##
import matplotlib.pyplot as plt
  
 ## creation d'un dataframe our stocker le nom et l'age ##
df = pd.DataFrame({
    'Name': ['awa', 'diarra', 'sire'],
    'Age': [25, 38, 19]
     })
  
 ## faire un gaphe en barr de ces deux variables ##
df.plot(x="Name", y="Age", kind="bar")
 
 #######pour l'histogramme  consommation de fruits(mangue=M, orange=O, Pomme=P #####
  
### importation des libraries####
 
 import matplotlib.pyplot as plt
  import pandas as pd
  import numpy as np
 
df4 = pd.DataFrame({'M': np.random.randn(1000) + 1,
                    'O': np.random.randn(1000),
                    'P': np.random.randn(1000) - 1},
                           columns =['M', 'O', 'P'])
plt.figure()
 
df4.plot.hist(alpha = 0.5)
plt.show()
   
   C_4) seaborn
    Seaborn est une bibliothèque permettant de créer des graphiques statistiques en Python.
    Cette bibliothèque est aussi performante que Matplotlib, mais apporte une simplicité et des fonctionnalités inédites. 
    Elle permet d’explorer et de comprendre rapidement les données.
    Les graphiques relationnels sont utilisés pour comprendre les relations entre deux variables, tandis que les graphiques 
    catégoriques permettent de visualiser des variables classées par catégorie.
    Quels sont les avantages de Seaborn ?
    La bibliothèque Seaborn offre plusieurs avantages majeurs. Elle fournit différents types de visualisations. Sa syntaxe est réduite,
    et elle propose des thèmes par défaut très attrayants. Il s’agit d’un outil idéal pour la visualisation statistique. 
    On l’utilise pour résumer les données dans les visualisations et la distribution des données.
    En outre, Seaborn est mieux intégré que Matplotlib pour travailler avec les data frames de Pandas. Enfin, il s’agit d’une 
    extension de Matplotlib pour créer de beaux graphiques à l’aide de Python grâce à un ensemble de méthodes plus directes.

                                Seaborn vs Matplotlib : lequel utiliser ?
    Matplotlib et Seaborn sont les deux outils Python les plus populaires pour la Data Visualization. Chacun présente des avantages et des inconvénients.
    On utilise principalement Matplotlib pour les tracés de graphiques basiques, tandis que Seaborn propose de nombreux thèmes par défaut et une vaste variété
    de schémas pour la visualisation de statistiques.
    En outre, Seaborn automatise la création de figures multiples. C’est un avantage, même si cela peut mener à des problèmes 
    d’utilisation de mémoire vie. Un autre atout de Seaborn est l’intégration renforcée avec Pandas et ses Data Frames, même si Matplotlib est aussi intégré 
    avec Pandas et NumPy.
    En revanche, Matplotlib offre une flexibilité accrue en termes de customisation et des performances parfois supérieures. Il peut donc s’agir d’une 
    meilleure option dans certaines situations.
    De manière générale, Seaborn est le meilleur choix d’outil de DataViz pour des visualisations de données statistiques. En revanche,
    Matplotlib répond mieux aux besoins en customisation. pour installer le module seaborn 
    utiliser la commande pip pour installer ce paquet depuis le terminal de commande. Cette commande est utilisée comme installateur de package en Python. 
    Nous pouvons installer le package seaborn en exécutant la commande ci-dessous.
    pip install seaborn
    quelques commandes avec Seaborn
    Seaborn nous fournit aussi des fonctions pour des graphiques utiles pour l'analyse statistique. Par exemple, la fonction distplot  permet non seulement de 
    visualiser l'histogramme d'un échantillon, mais aussi d'estimer la distribution dont l'échantillon est issu.
    sns.distplot(y, kde=True);
    Imaginons que nous voulons travailler sur un ensemble de données provenant du jeu de données "Iris", qui contient des mesures de la longueur et la
    largeur des sépales et des pétales de trois espèces d'iris. C'est un jeu de données très souvent utilisé pour se faire la main sur des problèmes de 
    machine learning.
    iris = sns.load_dataset("iris")
    iris.head()
   Pour voir les relations entre ces caractéristiques, on peut faire des graphiques par paire avec la commande pairplot:

   sns.pairplot(iris, hue='species', height=2.5);

  D) CREATION DE MODULES
   Dans cette partie, on parlera de la creation de son propre module sous python . En Python, la création d'un module est très simple. Il suffit d'écrire un 
   ensemble de fonctions (et/ou de constantes) dans un fichier, puis d'enregistrer ce dernier avec une extension .py (comme n'importe quel script Python). 
   À titre d'exemple, nous allons créer un module simple que nous enregisterons sous le nom message.py :
"""Module inutile qui affiche des messages :-)."""

   DATE = 16092008


   def bonjour(nom):
     """Dit Bonjour."""
     return "Bonjour " + nom


  def hello(nom):
     """Dit Hello."""
     return "Hello " + nom

  2)J'ai créer un package qvec deux autres modules qui se nomment: module1.py et module2.py
   module1.py a deux fonctions, l''une nous permet de calculer le carré d''un nombre donné
   l''autre nous permet de calculer le cube d''un nombre donné :

   def carre(valeur):
    resultat = valeur**2
    return resultat

   def cube(valeur):
    resultat = valeur**3
    return resultat


  module2.py est une classe qui nous permet d''obtenir le nom, la marque, le prix et l''état d''un voiture.
  class Voiture:
    def __init__(self,name,marque,prix,état):
        self.name=name
        self.marque=marque
        self.prix=prix
        self.état=état


    def getPrix(self):
        print("le prix de "+self.marque+ "est" + self.prix)

    def getName(self):
        print("cette voiture est un "+self.name)    

  ET maintenant j'ai créé un fichier applition.py pour utiliser les fonctons de ces modules
  
  import package1.module1 as pm1
  import package1.module2 as pm2
  b=7
  print("le carré de b est",pm1.carre(b))

  c=5
  u=pm1.cube(5)
  print("le cube de c est",u)


  V=pm2.Voiture('mercedes','Dakar KAR','7 million','neuf')
  print(V.name)
  print(V.prix)
  print(V.état)
  print(V.marque) 


  
E)ACCESSIBILITE DE MON MODULE PAR LES AUTRES UTILISATEURS SANS L'ENVOIE DES FICHIERS
 Grâce à des platformes comme GitHub où on peut déposer des packages, des fichiers d'instructions, de classe etc(module.py)
Une foi le lien de notre GitHub partagé, les autres peuvent avoir accès à notre module créé et déposé.
Ainsi, ils peuvent installer depuis leur terminal en utilisant le lien GitHub avec la commande pip install.  
  
III) CONCLUSION
En guise de synthèse, on a pu exploiter beaucoups de modules permettant de faire une exploitation des données de facon général. La rédaction de cet article, nous a permis de
 mieux comprendre les fonctions ainsi que les classes.

