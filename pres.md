# Présentation du Concours

## Objectif
Le but de la compétition est de **prédire l'identité d'un utilisateur de logiciel** à partir de traces d'utilisation collectées.

---

## Dates Clés
- **Début :** il y a 4 jours
- **Fin :** dans 3 mois

---

## Description

**Pouvez-vous prédire qui utilise mon logiciel ?**

La société **Infologic** édite et intègre des solutions logicielles pour l’agroalimentaire depuis 1982. Pour assurer l'évolution et la maintenance de ses logiciels, elle s’appuie sur des tests continus effectués par des équipes internes utilisant divers profils de test. Après chaque incident ou problème rencontré, les testeurs doivent reporter leur utilisation à partir de leur vrai profil, ce qui rend certaines manipulations longues et fastidieuses.

L'objectif de ce concours est donc de **déterminer s'il est possible d'identifier un utilisateur à partir de ses traces d'utilisation (actions sur le logiciel)**, en construisant un modèle de classification basé sur des techniques de machine learning.

Un tel modèle permettrait aussi de détecter des tentatives d’intrusion, par exemple quand un pirate cherche à se faire passer pour un utilisateur légitime.

Vous disposez donc d’un vaste ensemble de données de traces logicielles, à partir duquel :
1. Vous devez créer des **features** et entraîner un modèle.
2. Vous appliquez le modèle sur les données de test.
3. Vous soumettez vos prédictions.

*Bonne chance et bon apprentissage !*

---

## Évaluation

La métrique utilisée pour ce concours est le **score F1 moyen**.

- Le F1-score, largement utilisé dans la recherche d’informations, combine **précision** (p) et **rappel** (r) :
  - **Précision =** vrais positifs / (vrais positifs + faux positifs)
  - **Rappel =** vrais positifs / (vrais positifs + faux négatifs)
  - **F1-score =** (2 × p × r) / (p + r)

Un bon modèle devra maximiser à la fois la précision et le rappel. Il vaut mieux avoir une performance équilibrée que d’exceller dans une métrique au détriment de l’autre.

---

## Description des Données

### Fichiers disponibles

- **TRAIN.csv** : jeu d’apprentissage, avec les traces *étiquetées* (utilisateur connu)
- **TEST.csv** : jeu de test, traces *non étiquetées* (à prédire)
- **SAMPLE_SUBMISSION.csv** : exemple de fichier de soumission attendu

---

### Structure des Données (TRAIN.csv & TEST.csv)

Chaque ligne du fichier correspond aux actions d’un utilisateur durant une session.

#### Exemple
```
sph,Firefox,Exécution d'un bouton(fr.infologic.core.client.modules.web.CopiloteWebSharedController)<DEFAUT>$AC$,t5,Exécution d'un bouton,t10
```

#### Détail des champs

- **util** : ID unique de l’utilisateur (trigramme, seulement dans TRAIN.csv)
- **navigateur** : navigateur utilisé (Firefox, Google Chrome, Opera, Microsoft Edge)
- **Actions** : séquence d’actions effectuées par l’utilisateur, par exemple :
  - `"Création d'un écran"`, `"Double-clic"`, `"Chainage"`, etc.
  - Nouvelle fenêtre : mention entre parenthèses, ex :  
    `Création d'un écran(infologic.crm.modules.CRM_ANNUAIRE.AnnuaireController)`
  - Changement de configuration d’écran : entre balises, ex :  
    `Création d'un écran(infologic.core.gui.controllers.BlankController)<ACCUEIL_INST>`
  - Travail sur une fiche : chaîne d’identification entre dollars, ex :
    `Saisie dans un champ(MAINT)<DEF_03/24>$GP$`
  - Modification : chiffre `1` ajouté à la fin, ex :  
    `Saisie dans un champ1`
  - Temporalité : champs `tXX` indiquant des fenêtres temporelles (XX = nombre de secondes).  
    Par exemple :  
    - Toutes les actions avant `t20` ont été faites dans les 20 premières secondes.
    - Les actions entre `t20` et `t30` concernent l'intervalle [20;30], etc.
  - **Note** : Le nombre de fenêtres temporelles et d’actions varie selon les lignes, en fonction du rythme de chaque utilisateur et de la durée des sessions.

---

### À retenir
Après avoir lu cette description, vous devez savoir :
- **De quels fichiers ai-je besoin ?**
- **Quel format de données vais-je rencontrer ?**
- **Que dois-je prédire ?**
- **Quels acronymes ou structures importantes vais-je trouver ?**

**Si ce n’est pas clair, demandez à votre professeur !**