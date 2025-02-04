# Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuration du style des visualisations
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# -----------------------------
# 1. Chargement et préparation des données
# -----------------------------

# Charger le jeu de données
df = pd.read_excel("data/database.xlsx")

# Calcul de l'IMC (Indice de Masse Corporelle)
# IMC = Poids (kg) / (Taille (m))^2 ; conversion de la taille de cm à m
df['IMC'] = df['Poids (kg)'] / ((df['Taille (cm)'] / 100) ** 2)

# Création des classes d'âge
bins_age = [0, 18, 35, 50, 65, np.inf]
labels_age = ['0-18', '19-35', '36-50', '51-65', '65+']
df['Classe_age'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age, right=False)

# Création des classes d'IMC selon des seuils standards (OMS)
bins_imc = [0, 18.5, 25, 30, np.inf]
labels_imc = ['Insuffisance pondérale', 'Normale', 'Surpoids', 'Obésité']
df['Classe_IMC'] = pd.cut(df['IMC'], bins=bins_imc, labels=labels_imc)

# Affichage d'un aperçu et des informations
print("Aperçu des 5 premières lignes :")
print(df.head())
print("\nInformations sur les colonnes :")
df.info()
print("\nNombre de valeurs manquantes par colonne :")
print(df.isnull().sum())
print("\nStatistiques descriptives pour les variables numériques :")
print(df.describe())

# -----------------------------
# 2. Visualisations des données
# -----------------------------

# Graphique 1 : Répartition des individus suivant le genre
plt.figure()
sns.countplot(data=df, x='Sexe', palette="Set2")
plt.title("Répartition des individus selon le Genre")
plt.xlabel("Sexe")
plt.ylabel("Nombre d'individus")
plt.show()

# Graphique 2 : Répartition des individus suivant les classes d’IMC
plt.figure()
sns.countplot(data=df, x='Classe_IMC', palette="Set3", order=labels_imc)
plt.title("Répartition des individus par classes d'IMC")
plt.xlabel("Classe d'IMC")
plt.ylabel("Nombre d'individus")
plt.show()

# Graphique 3 : Obésité (proportion d'individus en obésité)
obesite_count = df['Classe_IMC'].value_counts().get('Obésité', 0)
total_count = len(df)
plt.figure()
plt.pie([obesite_count, total_count - obesite_count],
        labels=['Obésité', 'Non-Obésité'], autopct='%1.1f%%',
        colors=["red", "lightgray"])
plt.title("Proportion d'Obésité")
plt.show()

# Graphique 4 : Répartition des individus suivant les classes d’âge
plt.figure()
sns.countplot(data=df, x='Classe_age', palette="pastel", order=labels_age)
plt.title("Répartition des individus par classes d'âge")
plt.xlabel("Classe d'âge")
plt.ylabel("Nombre d'individus")
plt.show()

# Graphique 5 : Répartition des individus suivant leur état matrimonial
plt.figure()
sns.countplot(data=df, x='Etat_matrimonial', palette="cool")
plt.title("Répartition des individus par état matrimonial")
plt.xlabel("État matrimonial")
plt.ylabel("Nombre d'individus")
plt.xticks(rotation=45)
plt.show()

# Graphique 6 : Infection au VHB
plt.figure()
sns.countplot(data=df, x='Infection_VHB', palette="bright")
plt.title("Répartition des individus selon l'infection au VHB")
plt.xlabel("Infection VHB")
plt.ylabel("Nombre d'individus")
plt.show()

# Graphique 7 : Fréquence de contact avec le sang des pédiatres mariés non transfusés
subset = df[
    (df['Categorie_professionnelle'].str.contains("Pédiatre", case=False, na=False)) &
    (df['Etat_matrimonial'].str.contains("Marié", case=False, na=False)) &
    (df['Transfusion_sanguine'].str.contains("Non", case=False, na=False))
]
plt.figure()
sns.countplot(data=subset, x='Contact_avec_le_sang', palette="viridis")
plt.title("Contact avec le sang chez les pédiatres mariés non transfusés")
plt.xlabel("Contact avec le sang")
plt.ylabel("Nombre d'individus")
plt.show()

# Graphique 8 : Répartition des individus suivant leur catégorie professionnelle
plt.figure()
sns.countplot(data=df, x='Categorie_professionnelle', palette="magma")
plt.title("Répartition par catégorie professionnelle")
plt.xlabel("Catégorie professionnelle")
plt.ylabel("Nombre d'individus")
plt.xticks(rotation=45)
plt.show()

# Graphique 9 : Répartition des individus suivant leur groupe sanguin
plt.figure()
sns.countplot(data=df, x='Groupe_sanguin', palette="coolwarm")
plt.title("Répartition par groupe sanguin")
plt.xlabel("Groupe sanguin")
plt.ylabel("Nombre d'individus")
plt.show()

# Graphique 10 : Relation entre l’infection au VHB et l’état matrimonial
plt.figure()
ct = pd.crosstab(df['Etat_matrimonial'], df['Infection_VHB'])
ct.plot(kind='bar', stacked=True, colormap='viridis')
plt.title("Infection VHB selon l'état matrimonial")
plt.xlabel("État matrimonial")
plt.ylabel("Nombre d'individus")
plt.legend(title='Infection VHB')
plt.show()

# Graphique 11 : Relation entre l’infection au VHB et les différentes classes d’IMC
plt.figure()
ct = pd.crosstab(df['Classe_IMC'], df['Infection_VHB'])
ct = ct.reindex(labels_imc)  # Afin d'assurer le bon ordre
ct.plot(kind='bar', stacked=True, colormap='plasma')
plt.title("Infection VHB selon la classe d'IMC")
plt.xlabel("Classe d'IMC")
plt.ylabel("Nombre d'individus")
plt.legend(title='Infection VHB')
plt.show()

# Graphique 12 : Relation entre l’infection au VHB et le contact avec le sang
plt.figure()
ct = pd.crosstab(df['Contact_avec_le_sang'], df['Infection_VHB'])
ct.plot(kind='bar', stacked=True, colormap='cividis')
plt.title("Infection VHB selon le contact avec le sang")
plt.xlabel("Contact avec le sang")
plt.ylabel("Nombre d'individus")
plt.legend(title='Infection VHB')
plt.show()

# Graphique 13 : Relation entre l’infection au VHB et le groupe sanguin
plt.figure()
ct = pd.crosstab(df['Groupe_sanguin'], df['Infection_VHB'])
ct.plot(kind='bar', stacked=True, colormap='Accent')
plt.title("Infection VHB selon le groupe sanguin")
plt.xlabel("Groupe sanguin")
plt.ylabel("Nombre d'individus")
plt.legend(title='Infection VHB')
plt.show()

# Graphique 14 : Relation entre l’infection au VHB et le sexe des individus
plt.figure()
ct = pd.crosstab(df['Sexe'], df['Infection_VHB'])
ct.plot(kind='bar', stacked=True, colormap='Set1')
plt.title("Infection VHB selon le sexe")
plt.xlabel("Sexe")
plt.ylabel("Nombre d'individus")
plt.legend(title='Infection VHB')
plt.show()

# Graphique 15 : Relation entre l’infection au VHB et la catégorie professionnelle
plt.figure()
ct = pd.crosstab(df['Categorie_professionnelle'], df['Infection_VHB'])
ct.plot(kind='bar', stacked=True, colormap='tab20')
plt.title("Infection VHB selon la catégorie professionnelle")
plt.xlabel("Catégorie professionnelle")
plt.ylabel("Nombre d'individus")
plt.xticks(rotation=45)
plt.legend(title='Infection VHB')
plt.show()

# Graphique 16 : Relation entre l’infection au VHB et le service hospitalier des professionnels de santé
plt.figure()
ct = pd.crosstab(df['Service_hospitalier'], df['Infection_VHB'])
ct.plot(kind='bar', stacked=True, colormap='Spectral')
plt.title("Infection VHB selon le service hospitalier")
plt.xlabel("Service hospitalier")
plt.ylabel("Nombre d'individus")
plt.xticks(rotation=45)
plt.legend(title='Infection VHB')
plt.show()

# Graphique 17 : Relation entre l’infection au VHB et l’ancienneté des individus (Années de pratique)
plt.figure()
sns.boxplot(x='Infection_VHB', y='Annees_pratique_hospitaliere', data=df, palette="Set2")
plt.title("Années de pratique selon l'infection VHB")
plt.xlabel("Infection VHB")
plt.ylabel("Années de pratique hospitalière")
plt.show()

# Graphique 18 : Relation entre l’infection au VHB et la vaccination
plt.figure()
ct = pd.crosstab(df['Vaccination'], df['Infection_VHB'])
ct.plot(kind='bar', stacked=True, colormap='winter')
plt.title("Infection VHB selon la vaccination")
plt.xlabel("Vaccination")
plt.ylabel("Nombre d'individus")
plt.legend(title='Infection VHB')
plt.show()

# Graphique 19 : Relation entre l’infection au VHB et la transfusion sanguine
plt.figure()
ct = pd.crosstab(df['Transfusion_sanguine'], df['Infection_VHB'])
ct.plot(kind='bar', stacked=True, colormap='autumn')
plt.title("Infection VHB selon la transfusion sanguine")
plt.xlabel("Transfusion sanguine")
plt.ylabel("Nombre d'individus")
plt.legend(title='Infection VHB')
plt.show()

# Graphique 20 : Relation entre l’infection au VHB et la protection des rapports sexuels
plt.figure()
ct = pd.crosstab(df['Rapports_sexuels_proteges'], df['Infection_VHB'])
ct.plot(kind='bar', stacked=True, colormap='spring')
plt.title("Infection VHB selon la protection des rapports sexuels")
plt.xlabel("Rapports sexuels protégés")
plt.ylabel("Nombre d'individus")
plt.legend(title='Infection VHB')
plt.show()

# -----------------------------
# 3. Visualisations complémentaires (exemples déjà présents)
# -----------------------------

# Histogramme de la distribution du Poids
plt.figure()
sns.histplot(df['Poids (kg)'].dropna(), kde=True, bins=20, color="skyblue")
plt.title("Distribution du Poids (kg)")
plt.xlabel("Poids (kg)")
plt.ylabel("Fréquence")
plt.show()

# Histogramme de la distribution de la Taille
plt.figure()
sns.histplot(df['Taille (cm)'].dropna(), kde=True, bins=20, color="lightgreen")
plt.title("Distribution de la Taille (cm)")
plt.xlabel("Taille (cm)")
plt.ylabel("Fréquence")
plt.show()

# Boxplots pour détecter les outliers sur les variables Poids et Taille
plt.figure()
sns.boxplot(data=df[['Poids (kg)', 'Taille (cm)']])
plt.title("Boxplots de Poids et Taille")
plt.show()

# Scatter plot pour observer la relation entre Taille et Poids selon le Sexe
plt.figure()
sns.scatterplot(x='Taille (cm)', y='Poids (kg)', data=df, hue='Sexe', palette="deep", s=100)
plt.title("Relation entre la Taille et le Poids selon le Sexe")
plt.xlabel("Taille (cm)")
plt.ylabel("Poids (kg)")
plt.legend(title='Sexe')
plt.show()

# Scatter plot avec régression linéaire
plt.figure()
sns.regplot(x='Taille (cm)', y='Poids (kg)', data=df, scatter_kws={'alpha':0.6}, line_kws={"color": "red"})
plt.title("Régression linéaire entre Taille et Poids")
plt.xlabel("Taille (cm)")
plt.ylabel("Poids (kg)")
plt.show()

# Matrice de corrélation pour les variables numériques
numerical_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numerical_cols].corr()
plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Matrice de corrélation des variables numériques")
plt.show()
