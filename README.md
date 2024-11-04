# 🧠 Finsight AI

Finsight AI est un outil puissant conçu pour automatiser et optimiser l'analyse financière des entreprises. En examinant les rapports financiers annuels, la plateforme génère des rapports personnalisés, comparant les données spécifiques de l’entreprise avec les points de référence du marché. Le résultat inclut des graphiques clairs qui mettent en avant les principaux indicateurs, tendances et performances, offrant aux parties prenantes des insights exploitables sur la position de l’entreprise par rapport aux standards de l'industrie.

## Technologies Utilisées

Ce projet a été développé en JavaScript (React) et Python avec la contribution des personnes suivantes :

- Hossam Moustafa - [@scriptmaze](https://github.com/scriptmaze)
- Arnaud Grandisson - [@ArnaudGrd](https://github.com/ArnaudGrd)
- Ghali Chraibi - [@ghalichraibi](https://github.com/ghalichraibi)
- Hubert Khouzam - [@HubertKhouzam](https://github.com/HubertKhouzam)

## Pré-requis

- **Node.js** et **npm** pour le front-end
- **Python 3.x** et **pip** pour le back-end
- Environnement Virtuel (recommandé pour gérer les dépendances Python)

## Setup pour le Back-End (Analyse de Données)

1. Naviguez vers le dossier serveur :
   ```bash
   cd serveur
   ```

2. Créez un environnement virtuel :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   # Pour Windows : `venv\Scripts\activate`
   ```

3. Installez les dépendances Python :
   ```bash
   pip install -r requirements.txt
   ```

### Initialisation du fichier `.env`

Dans le dossier serveur, créez un fichier nommé `.env` pour y stocker les clés AWS nécessaires à l'extraction de données. Ce fichier doit respecter la structure suivante :

```plaintext
BUCKET_NAME='votre-nom-de-bucket'
AWS_ACCESS_KEY_ID='votre-aws-access-key-id'
AWS_SECRET_ACCESS_KEY='votre-aws-secret-access-key'
REGION_NAME='votre-region-name'
```

4. Lancez le serveur Flask en local :
   ```bash
   python your-app-name.py
   ```
   Le serveur devrait maintenant être accessible à [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Setup pour le Front-End

1. Ouvrez un nouveau terminal et accédez au dossier du front-end :
   ```bash
   cd client
   ```

2. Installez les dépendances front-end :
   ```bash
   npm ci
   ```

3. Démarrez le front-end en local :
   ```bash
   npm start
   ```
   L’interface devrait être accessible à [http://localhost:3000](http://localhost:3000)

---

Finsight AI est maintenant prêt à l'utilisation !

## Dépannage

Si vous rencontrez des erreurs :

- Vérifiez les informations d'authentification AWS dans `.env` pour vous assurer qu'elles sont correctes.
- Assurez-vous que le serveur et le client sont lancés dans des terminaux séparés.
- Consultez les logs pour des messages d'erreur spécifiques, que ce soit dans le terminal ou la console du navigateur.

Avec Finsight AI, prenez des décisions informées basées sur des analyses financières précises et automatisées ! 📊
