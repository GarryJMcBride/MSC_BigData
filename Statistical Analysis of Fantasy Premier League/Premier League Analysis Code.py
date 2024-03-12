import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn import metrics
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cluster
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# premierleagueplayers = pd.read_csv("C://Users//Garry//Documents//MSC Advanced Computer Science//Big data//Assignment 1"
#                                    "//FPL_2018_19_Wk0.csv")

premierleagueplayers = pd.read_csv("Statistical Analysis of Fantasy Premier League\Data\FPL_2018_19_Wk7.csv")


# How many observations and columns are there?
print(premierleagueplayers.shape)


# Print the names of all of the columns
print(premierleagueplayers.columns)


""" GENERAL DESCRIPTIVE ANALYSIS OF THE VARIABLES """

""" Players in Teams Count """

# Players in Team count for week 0 (Bar Chart)
status = ('ARS', 'BHA', 'BOU', 'BUR', 'CAR', 'CHE', 'CRY', 'EVE', 'FUL', 'HUD', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW',
          'SOU', 'TOT', 'WAT', 'WHU', 'WOL')
y_pos = np.arange(len(status))
numbers = [25, 24, 23, 21, 24, 25, 23, 26, 14, 25, 24, 26, 23, 25, 22, 25, 22, 28, 26, 21]
plt.bar(y_pos, numbers, align='center', color=['#85BB65'], alpha=0.5)
plt.xticks(y_pos, status)
plt.ylabel('Count', fontsize=35)
plt.xlabel('Teams', fontsize=35)
plt.title('Players in Each Team Week 0', fontsize=35)
plt.show()

# # Players in Team count for week 7 (Bar Chart)
status = ('ARS', 'BHA', 'BOU', 'BUR', 'CAR', 'CHE', 'CRY', 'EVE', 'FUL', 'HUD', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW',
          'SOU', 'TOT', 'WAT', 'WHU', 'WOL')
y_pos = np.arange(len(status))
numbers = [25, 26, 24, 26, 28, 28, 27, 34, 26, 28, 28, 25, 24, 27, 28, 27, 27, 29, 29, 26]
plt.bar(y_pos, numbers, align='center', color=['#2B65EC'], alpha=0.5)
plt.xticks(y_pos, status)
plt.ylabel('Count', fontsize=35)
plt.xlabel('Teams', fontsize=35)
plt.title('Players in Each Team Week 7', fontsize=35)
plt.show()


"""Number of players in certain position in each team"""


# Position count in each team week 0 (grouped bar chart)
# set width of bar
barWidth = 0.20

# set height of bar
bars1 = [2, 3, 2, 2, 2, 2, 3, 2, 1, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3]
bars2 = [10, 8, 7, 7, 9, 9, 11, 8, 4, 8, 8, 9, 8, 12, 7, 8, 8, 11, 10, 9]
bars3 = [10, 8, 10, 8, 9, 12, 5, 12, 8, 11, 10, 10, 11, 10, 9, 10, 10, 9, 10, 6]
bars4 = [3, 5, 4, 4, 4, 2, 4, 4, 1, 4, 3, 4, 2, 2, 3, 4, 2, 4, 4, 3]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='GOAL KEEPER')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='DEFENCE')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='MIDFIELD')
plt.bar(r4, bars4, color='#2B65EC', width=barWidth, edgecolor='white', label='FORWARD')

# Add xticks on the middle of the group bars
plt.ylabel('Count', fontsize=35)
plt.xlabel('Club', fontsize=35)
plt.title('Positions in Each Team Week 0', fontsize=35)
plt.xticks([r + barWidth for r in range(len(bars1))], ['ARS', 'BHA', 'BOU', 'BUR', 'CAR', 'CHE', 'CRY', 'EVE', 'FUL',
                                                       'HUD', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'SOU', 'TOT', 'WAT',
                                                       'WHU', 'WOL'])

# Create legend & Show graphic
plt.legend()
plt.show()

# Position count in each team week 7 (grouped bar chart)
# set width of bar
barWidth = 0.20

# set height of bar
bars1 = [3, 3, 2, 4, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 2, 3]
bars2 = [9, 9, 8, 8, 9, 9, 11, 12, 9, 9, 10, 10, 8, 11, 10, 9, 8, 12, 10, 11]
bars3 = [10, 9, 10, 9, 12, 3, 8, 18, 11, 12, 12, 10, 11, 11, 10, 10, 3, 9, 12, 9]
bars4 = [3, 5, 4, 5, 4, 3, 5, 4, 3, 4, 3, 3, 2, 2, 5, 5, 2, 5, 5, 3]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='GOAL KEEPER')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='DEFENCE')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='MIDFIELD')
plt.bar(r4, bars4, color='#2B65EC', width=barWidth, edgecolor='white', label='FORWARD')

# Add xticks on the middle of the group bars
plt.ylabel('Count', fontsize=35)
plt.xlabel('Club', fontsize=35)
plt.title('Positions in Each Team Week 7', fontsize=35)
plt.xticks([r + barWidth for r in range(len(bars1))], ['ARS', 'BHA', 'BOU', 'BUR', 'CAR', 'CHE', 'CRY', 'EVE', 'FUL',
                                                       'HUD', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'SOU', 'TOT', 'WAT',
                                                       'WHU', 'WOL'])

# Create legend & Show graphic
plt.legend()
plt.show()


""" Number of players in certain position """

# Position Count Combined teams’ week 0 ( Grouped Bar chart)
premierleagueplayers['Position'].value_counts().nlargest(4).sort_values(ascending=True).plot.barh()
plt.ylabel('Position', fontsize=35)
plt.xlabel('Count', fontsize=35)
plt.title('Positions Combined from all Teams week 0', fontsize=35)
plt.show()


# Position Count Combined teams’ week 7 ( Grouped Bar chart)
premierleagueplayers['Position'].value_counts().nlargest(4).sort_values(ascending=True).plot.barh()
plt.ylabel('Position', fontsize=35)
plt.xlabel('Count', fontsize=35)
plt.title('Positions Combined from all Teams week 7', fontsize=35)
plt.show()

""" Averages for Columns """

# Column Averages week 0 (Table)

# Column Averages week 7 (Table)

""" Variables of the Data that are related """

# Variable Relation week 0(heat map)
plt.figure(figsize=(20, 15))
plt.title('Variable Relations week 0', fontsize=35)
corr = premierleagueplayers.corr()
sns.heatmap(corr)
plt.show()

# Variable Relation week 7(heat map)
plt.figure(figsize=(20, 15))
plt.title('Variable Relations week 7', fontsize=35)
corr = premierleagueplayers.corr()
sns.heatmap(corr)
plt.show()


""" IN-DEPTH ANALYSIS OF VARIABLES """

""" Teams Based on Creativity """

# Squad Creativity week 0 (Density Diagram teams individually)
teams = ['ARS', 'BHA', 'BOU', 'BUR', 'CHE', 'CRY', 'EVE',
         'HUD', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'SOU', 'TOT', 'WAT', 'WHU']

    # Iterate through the 20 Teams
for team in teams:
    # Subset to the team
    subset = premierleagueplayers[premierleagueplayers['Team'] == team]

    # Draw the density plot
    sns.distplot(subset['Creativity'], hist=False, kde=True,
                 kde_kws={'linewidth': 2},
                 label=team)

    # Plot formatting
plt.legend(prop={'size': 12}, title='Teams')
plt.title('Density Team Creativity Week 0', fontsize=35)
plt.xlabel('Creativity', fontsize=35)
plt.ylabel('Density', fontsize=35)
plt.show()

# Squad Creativity week 7 (Density Diagram teams individually)
teams = ['ARS', 'BHA', 'BOU', 'BUR', 'CAR', 'CHE', 'CRY', 'EVE',
         'FUL', 'HUD', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'SOU', 'TOT', 'WAT', 'WHU', 'WOL']

    # Iterate through the 20 Teams
for team in teams:
    # Subset to the team
    subset = premierleagueplayers[premierleagueplayers['Team'] == team]

    # Draw the density plot
    sns.distplot(subset['Creativity'], hist=False, kde=True,
                 kde_kws={'linewidth': 2},
                 label=team)

    # Plot formatting
plt.legend(prop={'size': 12}, title='Teams')
plt.title('Density Team Creativity Week 7', fontsize=35)
plt.xlabel('Creativity', fontsize=35)
plt.ylabel('Density', fontsize=35)
plt.show()

# Squads Creativity week 0 (Bar Chart teams Individually)
y_pos = (premierleagueplayers['Team'])
x_pos = (premierleagueplayers['Creativity'])
plt.bar(y_pos, x_pos, align='center', color=['#85BB65'], alpha=0.5)
plt.title('Bar Team Creativity Week 0', fontsize=35)
plt.xlabel('Team', fontsize=35)
plt.ylabel('Creativity', fontsize=35)
plt.show()

# Squads Creativity week 7 (Bar Chart teams individually)
y_pos = (premierleagueplayers['Team'])
x_pos = (premierleagueplayers['Creativity'])
plt.bar(y_pos, x_pos, align='center', color=['#2B65EC'], alpha=0.5)
plt.title('Bar Team Creativity Week 7', fontsize=35)
plt.xlabel('Team', fontsize=35)
plt.ylabel('Creativity', fontsize=35)
plt.show()



""" Teams based on Influence """

# Squads Influence week 0 (Density Diagram teams individually)
teams = ['ARS', 'BHA', 'BOU', 'BUR', 'CHE', 'CRY', 'EVE',
         'HUD', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'SOU', 'TOT', 'WAT', 'WHU']

    # Iterate through the 20 Teams
for team in teams:
    # Subset to the team
    subset = premierleagueplayers[premierleagueplayers['Team'] == team]

    # Draw the density plot
    sns.distplot(subset['Influence'], hist=False, kde=True,
                 kde_kws={'linewidth': 2},
                 label=team)

    # Plot formatting
plt.legend(prop={'size': 12}, title='Teams')
plt.title('Density Team Influence Week 0', fontsize=35)
plt.xlabel('Influence', fontsize=35)
plt.ylabel('Density', fontsize=35)
plt.show()

# Squads Influence week 7 (Density Diagram teams individually)
teams = ['ARS', 'BHA', 'BOU', 'BUR', 'CAR', 'CHE', 'CRY', 'EVE',
         'FUL', 'HUD', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'SOU', 'TOT', 'WAT', 'WHU', 'WOL']

    # Iterate through the 20 Teams
for team in teams:
    # Subset to the team
    subset = premierleagueplayers[premierleagueplayers['Team'] == team]

    # Draw the density plot
    sns.distplot(subset['Influence'], hist=False, kde=True,
                 kde_kws={'linewidth': 2},
                 label=team)

    # Plot formatting
plt.legend(prop={'size': 12}, title='Teams')
plt.title('Density Team Influence Week 7', fontsize=35)
plt.xlabel('Influence', fontsize=35)
plt.ylabel('Density', fontsize=35)
plt.show()

# Squads Influence week 0 (Bar Chart teams Individually)
y_pos = (premierleagueplayers['Team'])
x_pos = (premierleagueplayers['Influence'])
plt.bar(y_pos, x_pos, align='center', color=['#85BB65'], alpha=0.5)
plt.title('Bar Team Influence Week 0', fontsize=35)
plt.xlabel('Team', fontsize=35)
plt.ylabel('Influence', fontsize=35)
plt.show()

# Squads Influence week 7 (Bar Chart teams individually)
y_pos = (premierleagueplayers['Team'])
x_pos = (premierleagueplayers['Influence'])
plt.bar(y_pos, x_pos, align='center', color=['#2B65EC'], alpha=0.5)
plt.title('Bar Team Influence Week 7', fontsize=35)
plt.xlabel('Team', fontsize=35)
plt.ylabel('Influence', fontsize=35)
plt.show()


""" Teams based on Threat """

# Squads Threat week 0 (Density Diagram teams individually)
teams = ['ARS', 'BHA', 'BOU', 'BUR', 'CHE', 'CRY', 'EVE',
         'HUD', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'SOU', 'TOT', 'WAT', 'WHU']

    # Iterate through the 20 Teams
for team in teams:
    # Subset to the team
    subset = premierleagueplayers[premierleagueplayers['Team'] == team]

    # Draw the density plot
    sns.distplot(subset['Threat'], hist=False, kde=True,
                 kde_kws={'linewidth': 2},
                 label=team)

    # Plot formatting
plt.legend(prop={'size': 12}, title='Teams')
plt.title('Density Team Threat Week 0', fontsize=35)
plt.xlabel('Threat', fontsize=35)
plt.ylabel('Density', fontsize=35)
plt.show()

# Squads Threat week 7 (Density Diagram teams individually)
teams = ['ARS', 'BHA', 'BOU', 'BUR', 'CAR', 'CHE', 'CRY', 'EVE',
         'FUL', 'HUD', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'SOU', 'TOT', 'WAT', 'WHU', 'WOL']

    # Iterate through the 20 Teams
for team in teams:
    # Subset to the team
    subset = premierleagueplayers[premierleagueplayers['Team'] == team]

    # Draw the density plot
    sns.distplot(subset['Threat'], hist=False, kde=True,
                 kde_kws={'linewidth': 2},
                 label=team)

    # Plot formatting
plt.legend(prop={'size': 12}, title='Teams')
plt.title('Density Team Threat Week 7', fontsize=35)
plt.xlabel('Threat', fontsize=35)
plt.ylabel('Density', fontsize=35)
plt.show()

# Squads Threat week 0 (Bar Chart teams Individually)
y_pos = (premierleagueplayers['Team'])
x_pos = (premierleagueplayers['Threat'])
plt.bar(y_pos, x_pos, align='center', color=['#85BB65'], alpha=0.5)
plt.title('Bar Team Threat Week 0', fontsize=35)
plt.xlabel('Team', fontsize=35)
plt.ylabel('Threat', fontsize=35)
plt.show()

# Squads Threat week 7 (Bar Chart teams individually)
y_pos = (premierleagueplayers['Team'])
x_pos = (premierleagueplayers['Threat'])
plt.bar(y_pos, x_pos, align='center', color=['#2B65EC'], alpha=0.5)
plt.title('Bar Team Threat Week 7', fontsize=35)
plt.xlabel('Team', fontsize=35)
plt.ylabel('Threat', fontsize=35)
plt.show()


""" Teams based on ICT """

# Squads ICT week 0 (Density Diagram teams individually)
teams = ['ARS', 'BHA', 'BOU', 'BUR', 'CHE', 'CRY', 'EVE',
         'HUD', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'SOU', 'TOT', 'WAT', 'WHU']

    # Iterate through the 20 Teams
for team in teams:
    # Subset to the team
    subset = premierleagueplayers[premierleagueplayers['Team'] == team]

    # Draw the density plot
    sns.distplot(subset['ICT'], hist=False, kde=True,
                 kde_kws={'linewidth': 2},
                 label=team)

    # Plot formatting
plt.legend(prop={'size': 12}, title='Teams')
plt.title('Density Team ICT Week 0', fontsize=35)
plt.xlabel('ICT', fontsize=35)
plt.ylabel('Density', fontsize=35)
plt.show()

# Squads ICT week 7 (Density Diagram teams individually)
teams = ['ARS', 'BHA', 'BOU', 'BUR', 'CAR', 'CHE', 'CRY', 'EVE',
         'FUL', 'HUD', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'SOU', 'TOT', 'WAT', 'WHU', 'WOL']

    # Iterate through the 20 Teams
for team in teams:
    # Subset to the team
    subset = premierleagueplayers[premierleagueplayers['Team'] == team]

    # Draw the density plot
    sns.distplot(subset['ICT'], hist=False, kde=True,
                 kde_kws={'linewidth': 2},
                 label=team)

    # Plot formatting
plt.legend(prop={'size': 12}, title='Teams')
plt.title('Density Team ICT Week 7', fontsize=35)
plt.xlabel('ICT', fontsize=35)
plt.ylabel('Density', fontsize=35)
plt.show()

Squads ICT week 0 (Bar Chart teams Individually)
y_pos = (premierleagueplayers['Team'])
x_pos = (premierleagueplayers['ICT'])
plt.bar(y_pos, x_pos, align='center', color=['#85BB65'], alpha=0.5)
plt.title('Bar Team ICT Week 0', fontsize=35)
plt.xlabel('Team', fontsize=35)
plt.ylabel('ICT', fontsize=35)
plt.show()

# Squads ICT week 7 (Bar Chart teams individually)
y_pos = (premierleagueplayers['Team'])
x_pos = (premierleagueplayers['ICT'])
plt.bar(y_pos, x_pos, align='center', color=['#2B65EC'], alpha=0.5)
plt.title('Bar Team ICT Week 7', fontsize=35)
plt.xlabel('Team', fontsize=35)
plt.ylabel('ICT', fontsize=35)
plt.show()

"""Unsupervised Method"""

premierleagueplayers.head()
# print(premierleagueplayers.head())

premierleagueplayers = premierleagueplayers.drop(columns=['Name', 'Team', 'Position'])
# print(premierleagueplayers)

X = premierleagueplayers.values[:, 4:7]
Y = premierleagueplayers.values[:, 3]
# print(X)
# print(Y)

scaled_premierleagueplayers = scale(X)
# print(scaled_premierleagueplayers)

# General clustering
from sklearn import cluster
from sklearn.preprocessing import LabelEncoder

n_samples, n_features = scaled_premierleagueplayers.shape
n_digits = len(np.unique(Y))
Y2 = LabelEncoder().fit_transform(Y)
aff = ["euclidean", "l1", "l2", "manhattan", "cosine"]
link = ["ward", "complete", "average"]
for a in aff:
    for l in link:
        if (l == "ward" and a != "euclidean"):
            continue
        else:
            print(a, l)
            model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage=l, affinity=a)
            model.fit(scaled_premierleagueplayers)
            print(metrics.silhouette_score(scaled_premierleagueplayers, model.labels_))
            print(metrics.completeness_score(Y2, model.labels_))
            print(metrics.homogeneity_score(Y2, model.labels_))

# this is  Hierarchical Clustering
# # try to get socre between 0-1,1 closer is better
from sklearn import cluster
from sklearn.preprocessing import LabelEncoder
n_samples, n_features = scaled_premierleagueplayers.shape
n_digits = len(np.unique(Y))
Y2 = LabelEncoder().fit_transform(Y)
# go online to find possible arguments for average and cosine
model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage="average", affinity="cosine")
model.fit(scaled_premierleagueplayers)
print(Y2)
print(model.labels_)
print(metrics.silhouette_score(scaled_premierleagueplayers, model.labels_))
print(metrics.completeness_score(Y2, model.labels_))
print(metrics.homogeneity_score(Y2, model.labels_))

# Denogram
from scipy.cluster.hierarchy import dendrogram, linkage

model = linkage(premierleagueplayers, 'ward')
Y2 = LabelEncoder().fit_transform(Y)
plt.figure()
plt.title('Hierarchical Clustering Dendrogram', fontsize=35)
plt.xlabel('sample index', fontsize=35)
plt.ylabel('distance', fontsize=35)
dendrogram(model, leaf_rotation=90., leaf_font_size=8.,)
plt.show()


"""Supervised Method (See Supervised Methods code file)"""


""" Positions based on ICT with goal scoring, assists, yellow & red cards """

# Player Creativity relation to Assits
plt.scatter(premierleagueplayers['Assists'], premierleagueplayers['Creativity'])
plt.title('Team ICT Week 0', fontsize=35)
plt.xlabel('Teams', fontsize=35)
plt.ylabel('ICT', fontsize=35)
plt.show()

# Player Threat relation to Goals Conceded
plt.scatter(premierleagueplayers['Goals_conceded'], premierleagueplayers['Threat'])
plt.title('Team ICT Week 0', fontsize=35)
plt.xlabel('Teams', fontsize=35)
plt.ylabel('ICT', fontsize=35)
plt.show()

# Player Threat related to Goals Scored
plt.scatter(premierleagueplayers['Goals_scored'], premierleagueplayers['Threat'])
plt.title('Team ICT Week 0', fontsize=35)
plt.xlabel('Teams', fontsize=35)
plt.ylabel('ICT', fontsize=35)
plt.show()

# Player Influence relation to Minutes Played
plt.scatter(premierleagueplayers['Minutes'], premierleagueplayers['Influence'])
plt.title('Team ICT Week 0', fontsize=35)
plt.xlabel('Teams', fontsize=35)
plt.ylabel('ICT', fontsize=35)
plt.show()




