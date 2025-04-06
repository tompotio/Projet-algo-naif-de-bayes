import numpy as np
import os
import math
import re
import pickle

epsilon = .1

# ======================================================================================
# 								ALGORITHME NAIF DE BAYES
# ======================================================================================

'''
	@brief	Fonction qui charge un dictionnaire dans un tableau (liste python) de
	mots à partir d'un fichier texte donné en paramètre. Un 

	@param fichier : Fichier texte.

	@return Dictionnaire chargé.
'''
def lireMail(fichier, dictionnaire : list):
	try:
		with open(fichier, "r", encoding="utf-8", errors="ignore") as file:
			texte = file.read().lower() # Lit le fichier et met tout en lowercase
	except Exception as ex:
		print(f"Erreur lors de la lecture de {fichier} : {ex}")
		return np.zeros(len(dictionnaire), dtype=bool)

	# pre-traitement sur texte 
	texte = re.sub(r"['’]", '', texte)

	# On garde que les mots d'au moins 3 lettres
	mots = re.findall(r'\b[a-z]{3,}\b', texte)

	x = np.zeros(len(dictionnaire), dtype=bool)

	for mot in mots:
		try:
			i = dictionnaire.index(mot)
			x[i] = True
		except:
			continue       

	return x


'''
	@brief	Fonction qui charge un dictionnaire dans un tableau (liste python) de
	mots à partir d'un fichier texte donné en paramètre. Un 

	@param fichier : Fichier texte.

	@return Dictionnaire chargé.
'''
def charge_dico(fichier):
	f = open(fichier, "r")
	mots = f.read().split("\n")
	f.close()

	return [str.lower(mot) for mot in mots  if len(mot) > 2] # Retire les mots avec moins de 3 lettres


'''
	@brief	Fonction d'apprentissage d'une loi binomiale a partir des fichiers d'un dossier

	@param dossier : Chemin du dossier contenant les données à apprendre.
	@param fichiers : Noms des fichiers des données à apprendre.
	@param dictionnaire : Mots connus sur lesquels apprendre.
			
	@return Un vecteur b de paramètres 
'''
def apprendBinomial(dossier, fichiers, dictionnaire):
	m = len(dictionnaire)
	N = len(fichiers)

	b = np.zeros(m)

	for fichier in fichiers:
		chemin_fichier = dossier + "/" + fichier
		x = lireMail(chemin_fichier, dictionnaire)  # vecteur binaire du mail
		b += x 

	global epsilon

	# Application du lissage de Laplace
	b = (b + epsilon) / (N + 2 * epsilon)  # Lissage : +1 au numérateur, +2 au dénominateur

	# b = b / N
	
	return b


'''
	@brief	Prédit si un mail représenté par un vecteur booléen x est un spam
	à partir du modèle de paramètres Pspam, Pham, bspam, bham.

	@param x : Vecteur booléens des mots apparaissant dans le mail.
	@param Pspam : Probabilité que le mail soit un SPAM.
	@param Pham : Probabilité que le mail soit un HAM.
	@param bspam : Vecteur des probabilités des mots appris étant susceptibles d'être dans un SPAM.
	@param bham : Vecteur des probabilités des mots appris étant susceptibles d'être dans un HAM.
			
	@return Le taux d'erreur 
'''
def prediction(x, Pspam, Pham, bspam, bham):
	"""
		Retourne True ou False.
	"""

	# Calcul des probabilités à l'aide du log
	logPspam = np.sum(
		x * np.log(bspam) + (1-x) * np.log( 1 - bspam)
	)

	logPham = np.sum(
		x * np.log(bham) + (1-x) * np.log( 1 - bham)
	)

	if np.isnan(logPham) or np.isnan(logPspam): 
		print("NaN")

	logPspam += np.log(Pspam)
	logPham += np.log(Pham)

	# Interprétation des sommes de log en probabilités entre ]0;1[ à l'aide de la fonction sigmoïde
	Pspam_x = 1 / (1 + np.exp(logPham - logPspam))
	Pham_x = 1 - Pspam_x
	
	# Checking SPAM ou HAM
	isSpam = (logPspam > logPham)

	return isSpam, Pspam_x, Pham_x


'''
	@brief	Teste le classifieur de paramètres Pspam, Pham, bspam, bhamsur 
	sur tous les fichiers d'un dossier étiquetés comme SPAM si isSpam et HAM sinon
		
	@return Le taux d'erreur 
'''

def test(dossier, isSpam, Pspam, Pham, bspam, bham):
	fichiers = os.listdir(dossier)
	nb_erreurs = 0
	total_mails = len(fichiers)
	
	for i in range(total_mails):
		fichier = fichiers[i]

		chemin_fichier = dossier + "/" + fichier		
		x = lireMail(chemin_fichier, dictionnaire)
		isSpam_pred, Pspam_x, Pham_x = prediction(x, Pspam, Pham, bspam, bham)

		if isSpam_pred != isSpam:
			nb_erreurs += 1

		output = f"SPAM numéro {i} :" if isSpam else f"HAM numéro {i} :"
		output += f" P(Y=SPAM | X=x) = {Pspam_x} P(Y=HAM | X=x) = {Pham_x}"

		if isSpam_pred and isSpam: 
			output += " => identifié comme un SPAM*" 
		elif isSpam_pred and not isSpam:
			output += " => identifié comme un SPAM *** erreur ***" 
		elif not isSpam_pred and isSpam:
			output += " => identifié comme un HAM *** erreur ***"
		else:
			output += " => identifié comme un HAM"

		print(output)

	return (nb_erreurs / total_mails)	


# ======================================================================================
# 									CLASSIFIEUR
# ======================================================================================

'''
	@brief Fonctionne exactement comme test mais à partir d'un classifieur 
	sous forme de structure plutôt que de toute la liste des paramètres.

	@param dossier : Dossier des mails à tester. 
	@param classifier :

	@return Le taux d'erreur.
'''
def testClassifieur(dossier, isSpam, classifieur):
	fichiers = os.listdir(dossier)
	nb_erreurs = 0
	total_mails = len(fichiers)

	Pspam, Pham, bspam, bham = (classifieur[k] for k in ["Pspam", "Pham", "bspam", "bham"])
	dictionnaire = classifieur["dictionnaire"]
	for i in range(total_mails):
		fichier = fichiers[i]

		chemin_fichier = dossier + "/" + fichier		
		x = lireMail(chemin_fichier, dictionnaire)
		isSpam_pred, Pspam_x, Pham_x = prediction(x, Pspam, Pham, bspam, bham)

		if isSpam_pred != isSpam:
			nb_erreurs += 1

		output = f"SPAM numéro {i} :" if isSpam else f"HAM numéro {i} :"
		output += f" P(Y=SPAM | X=x) = {Pspam_x} P(Y=HAM | X=x) = {Pham_x}"

		if isSpam_pred and isSpam: 
			output += " => identifié comme un SPAM*" 
		elif isSpam_pred and not isSpam:
			output += " => identifié comme un SPAM *** erreur ***" 
		elif not isSpam_pred and isSpam:
			output += " => identifié comme un HAM *** erreur ***"
		else:
			output += " => identifié comme un HAM"

		print(output)

	return (nb_erreurs / total_mails)


'''
	@brief Sauvegarde un classifieur.

	@param dossier : Chemin du dossier dans lequel enregistrer le classifieur.
	@param nom : Nom du fichier à enregistrer.
'''
def sauvegarderClassifieur(classifieur, dossier = "saves", nom = "classifieur.pkl"):
	if not os.path.exists(dossier):
		os.makedirs(dossier)
	chemin_fichier = os.path.join(dossier,nom)
	try:
		with open(chemin_fichier,"wb") as f:
			pickle.dump(classifieur,f)
			return 1
	except:
		print("Une erreur est suvrenue\nLe classifieur n'a pas pu être sauvegardé correctement.\n")
		return None


'''
	@brief Charge un classifieur et renvoie un objet classifieur, qui peut être ensuite utilisé.

	@dossier : Checmin du dossier dans lequel a été enregistré le classifieur.i

	@return Un classifieur.
'''
def chargerClassifieur(dossier = "saves", nom = "classifieur.pkl"):
	chemin_fichier = os.path.join(dossier,nom)
	if not os.path.exists(chemin_fichier):
		print(f"Erreur -> Aucun fichier de ce type : {nom}")
		return None
	else: 
		with open(chemin_fichier,"rb") as f:
			return pickle.load(f)


def updateClassifieur(isSpam, classifieur, dossiers = "baseapp/spam", dossier = "saves"):
	global epsilon 

	if classifieur == None:
		print("Erreur lors de la récupération du classifieur")
		return
	else:
		dictionnaire = classifieur["dictionnaire"]
		
		for Mail in mail : 
			chemin_mail = dossiers + "/" + Mail
			lecture_mail = lireMail(chemin_mail,dictionnaire)
			if isSpam:
				classifieur["mSpam"]+=1
				mSpam = classifieur["mSpam"]
				classifieur["bspam"] = (classifieur["bspam"]*(mSpam-1)+lecture_mail+epsilon) / (mSpam + 2 * epsilon)
			else:
				classifieur["mHam"]+=1
				mSpam = classifieur["mHam"]
				classifieur["bham"] = (classifieur["bham"]*(mHam-1)+lecture_mail+epsilon) / (mHam + 2 * epsilon)
				
		total = classifieur["mSpam"] + classifieur["mHam"]
		classifieur["Pspam"] = classifieur["mSpam"] / total
		classifieur["Pham"] = classifieur["mHam"] / total
		sauvegarde_classifieur = sauvegarderClassifieur(classifieur,"saves","classifieur.pkl")

		if sauvegarde_classifieur is not None:
			print("Sauvegarde Classifieur : Réussi")
			print("Le classifieur a bien été mis à jour !\n")
		else:
			print("Erreur lors de la sauvegarde du classifieur\n")


# ======================================================================================
# 										PROGRAMME
# ======================================================================================


def menu():
    print("\n===== FILTRE ANTI-SPAM : MENU PRINCIPAL =====")
    print("1. Sélectionner un classifieur existant")
    print("2. Créer un nouveau classifieur")
    print("3. Sauvegarder le classifieur courant")
    print("4. Lancer le test du classifieur")
    print("5. Supprimer un classifieur")
    print("6. Quitter")
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
    nom = input("Entrez le nom sous lequel sauvegarder le classifieur (exemple: monClassifieur.pkl) : ")
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
    print(f"Erreur de test sur {mSpam_test} SPAM : {spam_err_rate:.2f} %")
    print(f"Erreur de test sur {mHam_test} HAM : {ham_err_rate:.2f} %")
    print(f"Erreur de test globale sur {total_test} mails : {total_err_rate:.2f} %")


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
			print("Au revoir !")
			break
		else:
			print("Option non reconnue. Veuillez réessayer.")
