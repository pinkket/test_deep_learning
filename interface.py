from flask import Flask, request, render_template, jsonify
import face_recognition
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration pour le téléchargement des fichiers
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Créer le dossier uploads s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    # Votre fonction de prétraitement existante
    img = Image.fromarray(image)
    img = img.convert('RGB')
    return np.array(img)

@app.route('/', methods=['GET'])
def home():
    return render_template('interface.html')

@app.route('/compare', methods=['POST'])
def compare_faces():
    if 'known_image' not in request.files or 'unknown_image' not in request.files:
        return jsonify({'error': 'Les deux images sont requises'}), 400
    
    known_file = request.files['known_image']
    unknown_file = request.files['unknown_image']
    
    if known_file.filename == '' or unknown_file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if not (allowed_file(known_file.filename) and allowed_file(unknown_file.filename)):
        return jsonify({'error': 'Format de fichier non autorisé'}), 400
    
    try:
        # Sauvegarder les fichiers
        known_filename = secure_filename(known_file.filename)
        unknown_filename = secure_filename(unknown_file.filename)
        
        known_path = os.path.join(app.config['UPLOAD_FOLDER'], known_filename)
        unknown_path = os.path.join(app.config['UPLOAD_FOLDER'], unknown_filename)
        
        known_file.save(known_path)
        unknown_file.save(unknown_path)
        
        # Charger et prétraiter les images
        known_image = face_recognition.load_image_file(known_path)
        unknown_image = face_recognition.load_image_file(unknown_path)
        
        known_faces = face_recognition.face_encodings(known_image)
        unknown_faces = face_recognition.face_encodings(unknown_image)
        
        if len(known_faces) == 0:
            return jsonify({'result': 'Aucun visage détecté dans l\'image de référence'})
        elif len(unknown_faces) == 0:
            return jsonify({'result': 'Aucun visage détecté dans l\'image à comparer'})
        
        # Comparer les visages
        results = face_recognition.compare_faces([known_faces[0]], unknown_faces[0])
        distance = face_recognition.face_distance([known_faces[0]], unknown_faces[0])
        
        # Nettoyer les fichiers uploadés
        os.remove(known_path)
        os.remove(unknown_path)
        
        # Préparer la réponse
        interpretation = ""
        if distance[0] < 0.6:
            interpretation = "Très forte ressemblance"
        elif distance[0] < 0.7:
            interpretation = "Ressemblance probable"
        else:
            interpretation = "Faible ressemblance"
            
        return jsonify({
            'match': bool(results[0]),
            'distance': float(distance[0]),
            'interpretation': interpretation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)