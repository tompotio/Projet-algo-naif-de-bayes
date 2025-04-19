from interface import *

if __name__ == '__main__':
	classifieur_courant = None
	
	while True:
		choix = menu()
		if choix == "1":
			# Sélectionne un classifieur existant.
			classifieur_selectionné = selectionner_classifieur()
			if classifieur_selectionné:
				classifieur_courant = classifieur_selectionné
		elif choix == "2":
			# Crée un nouveau classifieur.
			classifieur_cree = creer_classifieur()
			if classifieur_cree: 
				classifieur_courant = classifieur_cree
		elif choix == "3":
			# Sauvegarde le classifieur courant.
			sauvegarder_classifieur_interface(classifieur_courant)
		elif choix == "4":
			# Lance le test du classifieur courant.
			lancer_test(classifieur_courant)
		elif choix == "5":
			# Supprime un classifieur.
			supprimer_classifieur()
		elif choix == "6":
			# Met à jour le classifieur.
			maj_classifieur(classifieur_courant)
		elif choix == "7":
			print("Au revoir !")
			break
		elif choix == "8":
			# Split un dataset en deux parties (apprentissage et test)
			split_dataset_interface()
		else:
			print("Option non reconnue. Veuillez réessayer.")