Utilisation :
Assurez-vous que Python 3 est installé sur votre système, ainsi que les bibliothèques requises : matplotlib, numpy, pandas, mip et openpyxl.

Dans certains cas, la bibliothèque pandas peut entrer en conflit avec d'autres bibliothèques (comme la mienne). Vous pouvez installer un environnement virtuel via conda.

Placez le fichier Excel ("AGRIBALYSE3.1_partie agriculture_conv_vf.xlsx") dans le même répertoire que le script.

Exécutez le script en exécutant python3 nom_du_script.py/python"Votre version" nom_du_script.py dans votre terminal, où "nom_du_script.py" est le nom du fichier script.

Commandes couramment utilisées par conda:

1) Vérifier quels sont les paquets installés

conda list
2) Pour voir quels environnements virtuels existent actuellement

conda env list 
conda info -e
3) Vérifier la mise à jour du conda actuel

conda update conda
3) Python pour créer un environnement virtuel

conda create -n votre_nom_env python=x.x
La commande anaconda crée un environnement virtuel avec la version python x.x et le nom votre_nom_env. Le fichier votre_nom_env se trouve dans le fichier envs du répertoire d'installation d'Anaconda.

4) Pour activer ou changer d'environnement virtuel

Ouvrez la ligne de commande et tapez python --version pour vérifier la version actuelle de python.

Linux : source activate your_env_nam
Windows : activate votre_nom_env
5) Pour installer des paquets supplémentaires dans l'environnement virtuel

conda install -n nom_de_votre_envoi [paquet]
6. fermer l'environnement virtuel (c'est-à-dire quitter l'environnement actuel et revenir à la version python par défaut dans l'environnement PATH)

désactiver nom_env
ou `activate root` pour revenir à l'environnement racine
Linux : source deactivate 
7. supprimer l'environnement virtuel

conda remove -n votre_nom_env --all