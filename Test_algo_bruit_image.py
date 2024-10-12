from PIL import Image, ImageDraw
import easygui
import os
from tqdm import tqdm
"""myimg = Image.open(easygui.fileopenbox())"""


# On ajoute du bruit

import numpy as np
import random

def bruit_gaussien(image, sigma):
    row, col = image.size
    channels = len(image.getbands())
    gauss = np.random.normal(0, sigma, (col, row, channels))
    noisy = np.array(image, dtype=np.float32)
    noisy = noisy + gauss
    noisy = np.clip(noisy, 0, 255)  # Pour s'assurer que les valeurs restent dans l'intervalle [0, 255]
    return Image.fromarray(noisy.astype(np.uint8))
def bruit_poivre_et_sel(image, proba):
    col, row = image.size
    noisy = np.array(image)
    for i in range(row):
        for j in range(col):
            rdn = random.random()
            if rdn < proba:
                noisy[i][j] = 0
            elif rdn > 1 - proba:
                noisy[i][j] = 255
    return Image.fromarray(noisy)
def inversion_X(image):
    return Image.fromarray(np.array(image)[:, ::-1])
def inversion_Y(image):
    return Image.fromarray(np.array(image)[::-1, :])
def rotation(image, angle):
    return image.rotate(angle)
def contrast(image, alpha):
    # Convertir l'image en array NumPy et multiplier par alpha
    contrasted = np.array(image, dtype=np.float32) * alpha
    # Clip les valeurs pour rester dans l'intervalle [0, 255] et convertir en uint8
    contrasted = np.clip(contrasted, 0, 255).astype(np.uint8)
    return Image.fromarray(contrasted)
def ajouter_forme(image):
    # Convertir l'image en mode RGBA pour permettre la transparence
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    
    # Obtenir les dimensions de l'image
    width, height = image.size
    


    # Choisir une forme aléatoire : carré, rectangle ou cercle
    """forme = random.choice(["carré", "rectangle", "cercle"])"""
    forme = random.choice(["rectangle", "cercle"])
    # Choisir une couleur aléatoire : noir, gris ou blanc
    couleur = random.choice([(0, 0, 0, 255), (128, 128, 128, 255), (255, 255, 255, 255)])
    
    # Définir les coordonnées de la forme
    x1 = random.randint(width // 4, width // 2)
    y1 = random.randint(height // 4, height // 2)
    x2 = x1 + random.randint(width // 10, width // 5)
    y2 = y1 + random.randint(height // 10, height // 5)
    
    """if forme == "carré":
        # Dessiner un carré
        side = min(x2 - x1, y2 - y1)
        draw.rectangle([x1, y1, x1 + side, y1 + side], fill=couleur)"""
    if forme == "rectangle":
        # Dessiner un rectangle
        draw.rectangle([x1, y1, x2, y2], fill=couleur)
    elif forme == "cercle":
        # Dessiner un cercle
        draw.ellipse([x1, y1, x2, y2], fill=couleur)
    
    return image
"""
def resize_and_center_image(image, target_shape):
    target_height, target_width = target_shape
    original_width, original_height = image.size

    # Calculer le ratio d'aspect
    aspect_ratio = original_width / original_height
    if target_width / target_height > aspect_ratio:
        new_height = target_height
        new_width = int(aspect_ratio * target_height)
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # Redimensionner l'image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Créer une nouvelle image avec la taille cible et un fond noir
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    # Calculer les positions pour centrer l'image
    top_left_x = (target_width - new_width) // 2
    top_left_y = (target_height - new_height) // 2
    # Coller l'image redimensionnée sur le fond noir
    new_image.paste(resized_image, (top_left_x, top_left_y))

    return new_image
"""

def crop_to_non_black(image):
    # Convertir l'image en tableau NumPy
    image_array = np.array(image)
    # Trouver les indices des pixels non noirs
    non_black_indices = np.where(np.any(image_array != [0, 0, 0], axis=-1))
    # Trouver les limites de recadrage
    top, bottom = np.min(non_black_indices[0]), np.max(non_black_indices[0])
    left, right = np.min(non_black_indices[1]), np.max(non_black_indices[1])
    # Recadrer l'image
    cropped_image = image.crop((left, top, right + 1, bottom + 1))
    return cropped_image

def resize_and_center_image(image, target_shape):
    target_height, target_width = target_shape
    
    # Recadrer l'image pour inclure uniquement les pixels non noirs
    cropped_image = crop_to_non_black(image)
    
    original_width, original_height = cropped_image.size

    # Calculer le ratio d'aspect
    aspect_ratio = original_width / original_height
    if target_width / target_height > aspect_ratio:
        new_height = target_height
        new_width = int(aspect_ratio * target_height)
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # Redimensionner l'image
    resized_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)

    # Créer une nouvelle image avec la taille cible et un fond noir
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    # Calculer les positions pour centrer l'image
    top_left_x = (target_width - new_width) // 2
    top_left_y = (target_height - new_height) // 2
    # Coller l'image redimensionnée sur le fond noir
    new_image.paste(resized_image, (top_left_x, top_left_y))

    return new_image

faire = 0.5
myimg = Image.open("C:/Users/joshu/Desktop/Cours IMT/2A/Commande entreprise/Base de données/ODIR/Images_train/Images/8_left.jpg")
"""#myimg_modified = bruit_gaussien(myimg, 80)
#myimg_modified = bruit_poivre_et_sel(myimg,0.05)
#myimg_modified = rotation(myimg, 30)
#myimg_modified = contrast(myimg, 1.5)
myimg_modified = ajouter_forme(myimg)"""

target_shape = (244, 244)
myimg = resize_and_center_image(myimg, target_shape)
#myimg.show()

def modif_aléatoire(img):
    # Liste des fonctions de modification aléatoire
    modifications = [bruit_gaussien, bruit_poivre_et_sel, inversion_X, inversion_Y, rotation, contrast, ajouter_forme]
    lambda_param = 1
    msg = "rien n'a été fait"
    poisson_variable = np.random.poisson(lambda_param)
    nb_modif = poisson_variable + 1
    n_max = len(modifications)
    # Créer un vecteur avec n valeurs de 1 et 7-n valeurs de 0
    if nb_modif == 0:
        return img, msg
    msg = "nombre de modifications : " + str(nb_modif) + " parmi lesquelles :"
    if nb_modif > n_max:
        nb_modif = n_max
    vector = np.array([1] * nb_modif + [0] * (7 - nb_modif))
    # Mélanger le vecteur
    np.random.shuffle(vector)
    # Appliquer les modifications aléatoires
    if vector[0] == 1 :
        img = modifications[0](img, np.random.uniform(5, 100))
        msg += "bruit gaussien,"
    if vector[1] == 1 :
        img = modifications[1](img, np.random.uniform(0.01, 0.1))
        msg += "bruit poivre et sel,"
    if vector[2] == 1 :
        img = modifications[2](img)
        msg += "inversion X,"
    if vector[3] == 1 :
        img = modifications[3](img)
        msg += "inversion Y,"
    if vector[4] == 1 :
        img = modifications[4](img, np.random.randint(0, 360))
        msg += "rotation,"
    if vector[5] == 1 :
        img = modifications[5](img, np.random.uniform(0.5, 2))
        msg += "contraste,"
    if vector[6] == 1 :
        img = modifications[6](img)
        msg += "ajout de forme,"
    return img, msg

if faire == 1:
    myimg_modified, msg = modif_aléatoire(myimg)
    print(msg)
    myimg_modified.show()



if faire == 0 :
    #d_shape = {}
    # Définir le chemin du dossier contenant les images
    path = "C:/Users/joshu/Desktop/Cours IMT/2A/Commande entreprise/Base de données/ODIR/Images_train/Images"
    # Lister tous les fichiers dans le dossier
    fichier = os.listdir(path)
    # Filtrer les fichiers pour ne garder que les images
    extensions_images = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    images = [f for f in fichier if f.lower().endswith(extensions_images)]
    print("Nombre d'images trouvées :", len(images))
    # Charger et afficher chaque image
    for image_nom in tqdm(images, desc="Traitement des images"):
        chemin_image = os.path.join(path, image_nom)
        image = Image.open(chemin_image)
        shape = np.shape(image)
        image = resize_and_center_image(image, target_shape)
        image_nom_sans_ext = os.path.splitext(image_nom)[0]
        
        chemin_sauvegarde = f"C:/Users/joshu/Desktop/Cours IMT/2A/Commande entreprise/Base de données/ODIR/Images_train/Images 244 244/{image_nom_sans_ext}_244.jpg"
        image.save(chemin_sauvegarde)
        
    print("Toutes les images ont été traitées et sauvegardées dans le dossier 'Images 244 244'.")