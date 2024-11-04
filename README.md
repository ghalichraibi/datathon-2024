# Finsight AI

Finsight AI est un outil performant conçu pour automatiser et optimiser l'analyse financière des entreprises. En examinant les rapports financiers annuels, la plateforme génère des rapports personnalisés, comparant les données spécifiques de l’entreprise avec les points de référence du marché. Le résultat inclut des graphiques clairs qui mettent en avant les principaux indicateurs, tendances et performances, permettant aux parties prenantes d’obtenir des repères exploitables sur la position de l’entreprise par rapport aux standards de l'industrie.

Cet outil à été créé en Javascript React et python

## Table des matières

- [Pré-requis](##Pré-requis)
- [Getting Started](#getting-started)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [License](#license)

---

## Pré-requis

- **Node.js** et **npm** pour le UI
- **Python 3.x** et **pip** pour le back-end
- **Virtual Environment** (recommandé pour gérer les dépendances python)


## Setup pour le back-end / l'analyse de données

```bash
cd serveur
```
- Création d'un environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate
## Windows : `venv\Scripts\activate`
```
- Installation des dépendances Python:
```bash
pip install -r requirements.txt
```

### Initialisation du fichier .env

Dans le dossier `serveur`, créer un nouveau fichier nommé `.env`. Ce fichier va contenir les clés AWS pour l'extraction de données.
Le fichier doit avoir la structure suivante:
```plaintext
BUCKET_NAME = 'your-bucket-name'
AWS_ACCESS_KEY_ID='your-aws-access-key-id'
AWS_SECRET_ACCESS_KEY='your-secret-access-key'
REGION_NAME='your-region-name'
```

Il est maintenant possible de déployer le serveur Flask localement:
```bash
python your-app-name.py
```
Le serveur devrait être déployé a l'adresse `http://127.0.0.1:5000`

## Setup pour le front-end

- Ouvrir un nouveau terminal et naviguer vers le front-end:
```bash
cd client
```

- Installer les dépendances du front-end
```bash
npm ci
```

- Déployer le front-end localement:
```bash
npm start
```
Le UI devrait être déployé a l'adresse `http://localhost:3000`

Finsight AI devrait maintenant être prêt à l'utilisation