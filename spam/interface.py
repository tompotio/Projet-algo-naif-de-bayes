import numpy as np
import os
import random
import shutil

from pathlib import Path
from bayes_classifier import *

def menu():
	print("\n===== FILTRE ANTI-SPAM : MENU PRINCIPAL =====")
	print("1. Sélectionner un classifieur existant")
	print("2. Créer un nouveau classifieur")
	print("3. Sauvegarder le classifieur courant")
	print("4. Lancer le test du classifieur")
	print("5. Supprimer un classifieur")
	print("6. Mettre à jour le classifieur")
	print("7. Quitter")
	print("8. Splitter un dataset (SPAM / HAM)")
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

    # Chargement du dictionnaire
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
    # Demande des chemins vers les dossiers SPAM et HAM
    spam_dir = input("Chemin vers le dossier contenant les SPAM : ").strip()
    ham_dir = input("Chemin vers le dossier contenant les HAM : ").strip()
    output_dir = input("Dossier de sortie (par défaut 'dataset') : ").strip() or "dataset"
    
    if not os.path.isdir(spam_dir) or not os.path.isdir(ham_dir):
        print("Un ou les deux dossiers n'existent pas. Vérifiez les chemins.")
        return

    # Demande des pourcentages pour le split global et pour SPAM/HAM dans chaque base
    try:
        train_ratio = float(input("Pourcentage de données pour l'apprentissage (par exemple 0.7 pour 70%) : "))
        if not (0 < train_ratio < 1):
            print("Le pourcentage d'apprentissage doit être compris entre 0 et 1.")
            return
    except ValueError:
        print("Entrée invalide pour le pourcentage d'apprentissage.")
        return

    try:
        spam_train_ratio = float(input("Pourcentage de SPAM dans l'apprentissage (par exemple 0.6 pour 60%) : "))
        ham_train_ratio = 1 - spam_train_ratio  # Complémentaire de SPAM
        if not (0 <= spam_train_ratio <= 1):
            print("Le pourcentage de SPAM dans l'apprentissage doit être compris entre 0 et 1.")
            return
    except ValueError:
        print("Entrée invalide pour le pourcentage de SPAM dans l'apprentissage.")
        return

    try:
        spam_test_ratio = float(input("Pourcentage de SPAM dans le test (par exemple 0.5 pour 50%) : "))
        ham_test_ratio = 1 - spam_test_ratio  # Complémentaire de SPAM
        if not (0 <= spam_test_ratio <= 1):
            print("Le pourcentage de SPAM dans le test doit être compris entre 0 et 1.")
            return
    except ValueError:
        print("Entrée invalide pour le pourcentage de SPAM dans le test.")
        return

    os.makedirs(os.path.join(output_dir, "train", "spam"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "ham"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "spam"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "ham"), exist_ok=True)

    def split_and_copy(source_dir, label, target_dir, target_ratio, ratio):
        fichiers = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        random.shuffle(fichiers)

        split_point = int(len(fichiers) * target_ratio)
        files_for_split = fichiers[:split_point]

        files_for_split_count = len(files_for_split)
        if files_for_split_count == 0:
            print(f"Aucun fichier à répartir pour {label}. Vérifiez les proportions.")
            return

        spam_count = int(files_for_split_count * ratio)
        ham_count = files_for_split_count - spam_count

        if spam_count > len([f for f in files_for_split if 'spam' in f.lower()]):
            print(f"Pas assez de fichiers SPAM dans {label} pour respecter le ratio demandé.")
            return
        if ham_count > len([f for f in files_for_split if 'ham' in f.lower()]):
            print(f"Pas assez de fichiers HAM dans {label} pour respecter le ratio demandé.")
            return

        for i, fichier in enumerate(files_for_split[:spam_count]):
            src = os.path.join(source_dir, fichier)
            dst = os.path.join(target_dir, "spam", fichier)
            shutil.copy2(src, dst)

        for i, fichier in enumerate(files_for_split[spam_count:]):
            src = os.path.join(source_dir, fichier)
            dst = os.path.join(target_dir, "ham", fichier)
            shutil.copy2(src, dst)

    split_and_copy(spam_dir, "spam", os.path.join(output_dir, "train"), train_ratio, spam_train_ratio)
    split_and_copy(ham_dir, "ham", os.path.join(output_dir, "train"), train_ratio, ham_train_ratio)

    split_and_copy(spam_dir, "spam", os.path.join(output_dir, "test"), 1 - train_ratio, spam_test_ratio)
    split_and_copy(ham_dir, "ham", os.path.join(output_dir, "test"), 1 - train_ratio, ham_test_ratio)

    print(f"\n Split terminé. Résultat dans : {Path(output_dir).resolve()}")