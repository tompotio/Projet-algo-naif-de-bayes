import numpy as np
import os
import math
import re
import pickle

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

def charge_dico(fichier):
	f = open(fichier, "r")
	mots = f.read().split("\n")
	f.close()

	return [str.lower(mot) for mot in mots  if len(mot) > 2] # Retire les mots avec moins de 3 lettres

def apprendBinomial(dossier, fichiers, dictionnaire):
	"""
	Fonction d'apprentissage d'une loi binomiale a partir des fichiers d'un dossier
	Retourne un vecteur b de paramètres 
	"""

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


def prediction(x, Pspam, Pham, bspam, bham):
	"""
		Prédit si un mail représenté par un vecteur booléen x est un spam
		à partir du modèle de paramètres Pspam, Pham, bspam, bham.
		Retourne True ou False.
	"""

	logPspam = np.sum(
		np.log(
			(bspam ** x) * ((1 - bspam) ** (1 - x))
		)
	)

	logPham = np.sum(
		np.log(
			(bham ** x) * ((1 - bham) ** (1 - x))
		)
	)

	logPspam += np.log(Pspam)
	logPham += np.log(Pham)
	
	return (logPspam > logPham)
	
def test(dossier, isSpam, Pspam, Pham, bspam, bham):
	"""
		Test le classifieur de paramètres Pspam, Pham, bspam, bham 
		sur tous les fichiers d'un dossier étiquetés 
		comme SPAM si isSpam et HAM sinon
		
		Retourne le taux d'erreur 
	"""

	fichiers = os.listdir(dossier)
	nb_erreurs = 0
	total_mails = len(fichiers)
	
	for fichier in fichiers:
		chemin_fichier = dossier + "/" + fichier		
		x = lireMail(chemin_fichier, dictionnaire)
		isSpam_prediction = prediction(x, Pspam, Pham, bspam, bham)

		if isSpam_prediction != isSpam:
			nb_erreurs += 1
		
		'''
		if isSpam_prediction and isSpam: 
			print("SPAM ", chemin_fichier, " identifié comme un SPAM" )
		elif isSpam_prediction and not isSpam:
			print("HAM ", chemin_fichier, " identifié comme un SPAM *** erreur ***" )
		elif not isSpam_prediction and isSpam:
			print("SPAM ", chemin_fichier, " identifié comme un HAM *** erreur ***" )
		else:
			print("HAM ", chemin_fichier, " identifié comme un HAM")
		'''

	return (nb_erreurs / total_mails)

############ programme principal ############

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
