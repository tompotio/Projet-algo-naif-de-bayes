import numpy as np
import os
import math
import re
import pickle

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

	epsilon = .1

	# Application du lissage de Laplace (epsilon = 1)
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

	'''
	logPspam = np.sum(
		np.log(
			(bspam**x)*((1-bspam)**(1-x))
		)
	)

	logPham = np.sum(
		np.log(
			(bham**x)*((1-bham)**(1-x))
		)
	)
	'''

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

def testClassifieur(dossier, isSpam, classifieur):
	pass

'''
	@brief Sauvegarde un classifieur.

	@param dossier : Chemin du dossier dans lequel enregistrer le classifieur.
	@param nom : Nom du fichier à enregistrer.
'''
def sauvegarderClassifieur(classifieur, dossier = "saves", nom = "classifieur.pkl"):
	try:
		with open(nom,"wb") as f:
			pickle.dump(classifieur,f)
	except:
		print("Une erreur est suvrenue\nLe classifieur n'a pas pu être sauvegardé correctement.\n")

'''
	@brief Charge un classifieur et renvoie un objet classifieur, qui peut être ensuite utilisé.

	@dossier : Checmin du dossier dans lequel a été enregistré le classifieur.i

	@return Un classifieur.
'''

'''
def chargerClassifieur(dossier = "saves", nom = "classifieur.pkl"):
	if not os.path.exists(nom):
		print(f"Erreur -> Aucun fichier de ce type : {nom}")
		return None
	else: open(nom,"rb") as f:

nomdosreturn pickle.load(f)er_spams =f 
'''

# ======================================================================================
# 										PROGRAMME
# ======================================================================================
			
dossier_spams = "baseapp/spam"
dossier_hams = "baseapp/ham"

fichiersspams = os.listdir(dossier_spams)
fichiershams = os.listdir(dossier_hams)

mSpam = len(fichiersspams)
mHam = len(fichiershams)
total = (mSpam + mHam)

# Chargement du dictionnaire:
dictionnaire = charge_dico("dictionnaire1000en.txt")
#print(dictionnaire)

# Apprentissage des bspam et bham:
print("apprentissage de bspam...")
bspam = apprendBinomial(dossier_spams, fichiersspams, dictionnaire)
print("apprentissage de bham...")
bham = apprendBinomial(dossier_hams, fichiershams, dictionnaire)

# Calcul des probabilités a priori Pspam et Pham:
Pspam = mSpam / total
Pham = mHam / total

# Calcul des erreurs avec la fonction test():

dossier_spams_test = "basetest/spam"
dossier_hams_test = "basetest/ham"

fichiersspams_test = os.listdir(dossier_spams_test)
fichiershams_test = os.listdir(dossier_hams_test)

mSpam_test = len(fichiersspams_test)
mHam_test = len(fichiershams_test)
total_test = (mSpam_test + mHam_test)

fichiersspams_test = os.listdir(dossier_spams_test)
fichiershams_test = os.listdir(dossier_hams_test)

spam_err_rate = test(dossier_spams_test, True, Pspam, Pham, bspam, bham) * 100
ham_err_rate = test(dossier_hams_test, False, Pspam, Pham, bspam, bham) * 100
total_err_rate = (((spam_err_rate * mSpam) + (ham_err_rate * mHam)) / total_test)

print("Erreur de test sur ", mSpam_test, " SPAM : ", spam_err_rate, " %")
print("Erreur de test sur ", mHam_test, " HAM : ", ham_err_rate, " %")
print("Erreur de test globale sur ", total_test, " mails : ", total_err_rate, " %")

classifieur = {
    "Pspam": Pspam,
    "Pham": Pham,
    "bspam": bspam,
    "bham": bham,
    "dictionnaire": dictionnaire,
    "mSpam": mSpam,
    "mHam": mHam
}