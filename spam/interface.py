import numpy as np
import os
import random
import shutil
import math

from pathlib import Path
from bayes_classifier import *

dossier_dicos = "dics"

def menu():
    print("\n===== FILTRE ANTI-SPAM : MENU PRINCIPAL =====")
    print("1. Sélectionner un classifieur existant")
    print("2. Créer un nouveau classifieur")
    print("3. Sauvegarder le classifieur courant")
    print("4. Lancer le test du classifieur")
    print("5. Supprimer un classifieur")
    print("6. Mettre à jour le classifieur")
    print("7. Splitter un dataset (SPAM / HAM)")
    print("8. Quitter")
    return input("Votre choix : ")


def lister_classifieurs(dossier="saves"):
    import os
    if not os.path.exists(dossier):
        print("Le dossier de sauvegarde n'existe pas.")
        return []
    fichiers = [f for f in os.listdir(dossier) if f.endswith(".pkl")]
    if not fichiers:
        print("Aucun classifieur n'a été trouvé.")
    else:
        for idx, nom in enumerate(fichiers):
            print(f"{idx+1}. {nom}")
    return fichiers


def selectionner_classifieur():
    fichiers = lister_classifieurs()
    if not fichiers:
        return None
    choix = input("Sélectionnez le numéro du classifieur à charger : ")
    try:
        idx = int(choix) - 1
        nom = fichiers[idx]
        # Charger le classifieur
        classifieur = chargerClassifieur(dossier="saves", nom=nom)
        print(f"Classifieur {nom} chargé.")
        return classifieur
    except (ValueError, IndexError):
        print("Choix invalide.")
        return None


def creer_classifieur():
    print("\n--- Création d'un nouveau classifieur ---")
    choix_base = input("Utiliser la base par défaut (tapez 'd') ou une base personnalisée (tapez 'p') ? ")
    if choix_base.lower() == 'd':
        dossier_spams = "baseapp/spam"
        dossier_hams = "baseapp/ham"
    elif choix_base.lower() == 'p':
        dossier_spams = input("Entrez le chemin vers le dossier des SPAM : ")
        dossier_hams = input("Entrez le chemin vers le dossier des HAM : ")
    else:
        print("Choix non reconnu. Utilisation de la base par défaut.")
        dossier_spams = "baseapp/spam"
        dossier_hams = "baseapp/ham"

    # Choix du dictionnaire
    os.makedirs(dossier_dicos, exist_ok=True)
    dicos = [f for f in os.listdir(dossier_dicos) if f.endswith(".txt")]

    print("\n--- Sélection du dictionnaire ---")
    if dicos:
        print("Dictionnaires disponibles :")
        for idx, nom in enumerate(dicos):
            print(f"{idx + 1}. {nom}")
        print(f"{len(dicos) + 1}. Importer un nouveau dictionnaire")

        choix_dico = input("Choisissez un dictionnaire (numéro) : ")
        try:
            idx = int(choix_dico) - 1
            # Choisit d'importer un nouveau dictionnaire ou bien d'utiliser un de ceux déjà là
            if idx == len(dicos):
                chemin_nouveau_dico = input("Chemin vers le nouveau dictionnaire : ").strip()
                if os.path.isfile(chemin_nouveau_dico):
                    nouveau_nom = os.path.basename(chemin_nouveau_dico)
                    shutil.copy2(chemin_nouveau_dico, os.path.join(dossier_dicos, nouveau_nom))
                    print(f"Dictionnaire '{nouveau_nom}' importé.")
                    dictionnaire = charge_dico(os.path.join(dossier_dicos, nouveau_nom))
                else:
                    print("Fichier introuvable. Utilisation du dictionnaire par défaut.")
                    dictionnaire = charge_dico("dictionnaire1000en.txt")
            else:
                dictionnaire = charge_dico(os.path.join(dossier_dicos, dicos[idx]))
        except (ValueError, IndexError):
            print("Choix invalide. Utilisation du dictionnaire par défaut.")
            dictionnaire = charge_dico("dictionnaire1000en.txt")
    else:
        print("Aucun dictionnaire trouvé dans le dossier. Utilisation du dictionnaire par défaut.")
        dictionnaire = charge_dico("dictionnaire1000en.txt")
    
    # Apprentissage sur les spams
    fichiers_spams = os.listdir(dossier_spams)
    print("Apprentissage des SPAM...")
    bspam = apprendBinomial(dossier_spams, fichiers_spams, dictionnaire)
    mSpam = len(fichiers_spams)

    # Apprentissage sur les hams
    fichiers_hams = os.listdir(dossier_hams)
    print("Apprentissage des HAM...")
    bham = apprendBinomial(dossier_hams, fichiers_hams, dictionnaire)
    mHam = len(fichiers_hams)

    total = mSpam + mHam
    Pspam = mSpam / total
    Pham = mHam / total

    # Constitution du classifieur sous forme de dictionnaire
    classifieur = {
        "Pspam": Pspam,
        "Pham": Pham,
        "bspam": bspam,
        "bham": bham,
        "dictionnaire": dictionnaire,
        "mSpam": mSpam,
        "mHam": mHam
    }
    print("Nouveau classifieur créé.")
    return classifieur


def sauvegarder_classifieur_interface(classifieur):
	if classifieur is None:
		print("Aucun classifieur n'est chargé pour sauvegarde.")
		return
	nom = input("Entrez le nom sous lequel sauvegarder le classifieur (exemple: monClassifieur) : ")
	nom = nom + ".pkl"
	if sauvegarderClassifieur(classifieur, dossier="saves", nom=nom):
		print(f"Classifieur sauvegardé sous {nom}.")
	else:
		print("Échec de la sauvegarde.")


def lancer_test(classifieur):
	if classifieur is None:
		print("Aucun classifieur n'est chargé pour effectuer un test.")
		return
		
	# Choix de la base de test
	choix_test = input("Utiliser la base de test par défaut (tapez 'd') ou une base personnalisée (tapez 'p') ? ")
	if choix_test.lower() == 'd':
		dossier_spams_test = "basetest/spam"
		dossier_hams_test = "basetest/ham"
	elif choix_test.lower() == 'p':
		dossier_spams_test = input("Entrez le chemin vers le dossier de test des SPAM : ")
		dossier_hams_test = input("Entrez le chemin vers le dossier de test des HAM : ")
	else:
		print("Choix non reconnu. Utilisation de la base de test par défaut.")
		dossier_spams_test = "basetest/spam"
		dossier_hams_test = "basetest/ham"

	fichiers_spams_test = os.listdir(dossier_spams_test)
	fichiers_hams_test = os.listdir(dossier_hams_test)
	mSpam_test = len(fichiers_spams_test)
	mHam_test = len(fichiers_hams_test)
	total_test = mSpam_test + mHam_test

	# Test sur spam et ham
	print("\nTest sur les SPAM:")
	spam_err_rate = testClassifieur(dossier_spams_test, True, classifieur) * 100
	print("\nTest sur les HAM:")
	ham_err_rate = testClassifieur(dossier_hams_test, False, classifieur) * 100

	total_err_rate = (((spam_err_rate * mSpam_test) + (ham_err_rate * mHam_test)) / total_test)
	print("\n===== RÉSULTATS DU TEST =====")
	print("Erreur de test sur ", mSpam_test, " SPAM : ", spam_err_rate, " %")
	print("Erreur de test sur ", mHam_test, " HAM : ", ham_err_rate, " %")
	print("Erreur de test globale sur ", total_test, " mails : ", total_err_rate, " %")


def supprimer_classifieur():
    fichiers = lister_classifieurs()
    if not fichiers:
        return
    choix = input("Entrez le numéro du classifieur à supprimer : ")
    try:
        idx = int(choix) - 1
        nom = fichiers[idx]
        chemin = os.path.join("saves", nom)
        os.remove(chemin)
        print(f"Classifieur {nom} supprimé.")
    except Exception as e:
        print("Erreur lors de la suppression :", e)


def maj_classifieur(classifieur):
    chemin = input("Veuillez renseigner le chemin absolu vers le fichier ou dossier de mails : ").strip()
    isSpam = input("Les mails sont-ils des spams ? (tapez 'y' ou 'n') : ").strip().lower()
    spam_flag = isSpam == 'y'

	# Fichier unique
    if os.path.isfile(chemin):
        return updateClassifieur(chemin, spam_flag, classifieur)
    
	# Dossier contenant plusieurs fichiers
    elif os.path.isdir(chemin):
        for nom_fichier in os.listdir(chemin):
            chemin_fichier = os.path.join(chemin, nom_fichier)
            if os.path.isfile(chemin_fichier):  # éviter les sous-dossiers
                classifieur = updateClassifieur(chemin_fichier, spam_flag, classifieur)
        return classifieur
    else:
        print("Chemin invalide. Veuillez fournir un fichier ou un dossier existant.")
        return classifieur


def split_dataset_interface():
    print("\n=== SPLIT DU DATASET ===")
    spam_dir = input("Chemin vers le dossier contenant les mails SPAM : ").strip()
    ham_dir = input("Chemin vers le dossier contenant les mails HAM : ").strip()
    output_dir = input("Dossier de sortie (par défaut 'dataset') : ").strip() or "dataset"

    if not os.path.isdir(spam_dir) or not os.path.isdir(ham_dir):
        print("Un ou les deux dossiers n'existent pas. Vérifiez les chemins.")
        return

    try:
        spam_ratio = float(input("Proportion des SPAM à utiliser pour l'entraînement (ex: 0.7 pour 70%) : "))
        ham_ratio = float(input("Proportion des HAM à utiliser pour l'entraînement (ex: 0.5 pour 50%) : "))
        if not (0 < spam_ratio < 1) or not (0 < ham_ratio < 1):
            print("Les ratios doivent être entre 0 et 1.")
            return
    except ValueError:
        print("Entrée invalide.")
        return

    # Crée les dossiers de sortie
    for subset in ['train', 'test']:
        for label in ['spam', 'ham']:
            os.makedirs(os.path.join(output_dir, subset, label), exist_ok=True)

    # Fonction de split et copie
    def split_and_copy(source_dir, label, ratio):
        fichiers = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        random.shuffle(fichiers)

        n_total = len(fichiers)
        n_train = math.floor(ratio * n_total)
        n_test = n_total - n_train

        train_files = fichiers[:n_train]
        test_files = fichiers[n_train:]

        for f in train_files:
            shutil.copy2(os.path.join(source_dir, f), os.path.join(output_dir, 'train', label, f))
        for f in test_files:
            shutil.copy2(os.path.join(source_dir, f), os.path.join(output_dir, 'test', label, f))

        print(f"{label.upper()} : {n_train} pour train, {n_test} pour test (total : {n_total})")

    # Split par classe avec les bons ratios
    split_and_copy(spam_dir, "spam", spam_ratio)
    split_and_copy(ham_dir, "ham", ham_ratio)

    print(f"\nSplit terminé. Résultat enregistré dans : {Path(output_dir).resolve()}")