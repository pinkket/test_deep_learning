import face_recognition
import os
from PIL import Image
import numpy as np

def preprocess_images(image_paths):
    """
    Prétraite plusieurs images en même temps
    Retourne une liste des images prétraitées
    """
    processed_images = []
    
    for image_path in image_paths:
        try:
            # Ouvrir l'image
            img = Image.open(image_path)
            
            # Convertir en RGB si nécessaire
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Redimensionner si l'image est trop grande
            max_size = 1600  # Taille maximale recommandée
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple([int(x * ratio) for x in img.size])
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convertir en array numpy
            img_array = np.array(img)
            processed_images.append(img_array)
            
            print(f"Prétraitement réussi pour {image_path}")
            
        except Exception as e:
            print(f"Erreur lors du prétraitement de {image_path}: {str(e)}")
            processed_images.append(None)
    
    return processed_images

# Chemins des images
image_paths = [
    '/Users/kettytouvoli/Desktop/M2/Ml_non_supervisé/test_deep_learning/Jasmine.jpg',
    '/Users/kettytouvoli/Desktop/M2/Ml_non_supervisé/test_deep_learning/Jasmine1.jpg'
]

# Vérifier l'existence des fichiers
for path in image_paths:
    if not os.path.exists(path):
        print(f"Erreur: Fichier non trouvé: {path}")
        exit()

# Prétraitement des images
processed_images = preprocess_images(image_paths)
known_image = processed_images[0]
unknown_image = processed_images[1]

# Check for None in processed_images
if any(img is None for img in processed_images):
    print("Erreur lors du prétraitement d'une ou plusieurs images")
    exit()

# ... existing code ...

# Face recognition
known_faces = face_recognition.face_encodings(known_image)
unknown_faces = face_recognition.face_encodings(unknown_image)

# Ajouter ce code pour comparer et afficher les résultats
if len(known_faces) == 0:
    print("Aucun visage détecté dans l'image de référence")
elif len(unknown_faces) == 0:
    print("Aucun visage détecté dans l'image à comparer")
else:
    # Comparer le premier visage trouvé dans chaque image
    results = face_recognition.compare_faces([known_faces[0]], unknown_faces[0])
    distance = face_recognition.face_distance([known_faces[0]], unknown_faces[0])
    
    print("\nRésultats de la reconnaissance faciale:")
    print(f"Match trouvé: {'Oui' if results[0] else 'Non'}")
    print(f"Distance (plus le chiffre est petit, plus la ressemblance est forte): {distance[0]:.2f}")
    
    # Interprétation de la distance
    if distance[0] < 0.6:
        print("Interprétation: Très forte ressemblance")
    elif distance[0] < 0.7:
        print("Interprétation: Ressemblance probable")
    else:
        print("Interprétation: Faible ressemblance")