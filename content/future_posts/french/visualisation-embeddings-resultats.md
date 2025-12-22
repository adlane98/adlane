---
title: "Visualisation et analyse des embeddings - Résultats"
date: 2025-12-01T19:53:33+05:30
draft: true
author: "Adlane Ladjal"
tags:
  - deep learning
  - ai
  - koleo loss
image: /images/koleo-loss-illustration.png
description: ""
toc: true
mathjax: true
---

Dans le précédent chapitre nous avons entraîné notre premier modèle siamois. Il fonctionne car la loss de validation décroit sur les premières epochs. De plus les métriques de validation s'améliorent d'epoch en epoch. Ce sont d'excellents signes que l'apprentissage s'est bien passé. Nous allons pouvoir commencer à analyser nos embeddings, d'une part avec la matrice de distances, et d'autre part à l'aide d'une ACP.

# Matrice de distances

On peut déjà comparer les distances entre `ancre` et `positive`, et celles entre `ancre` et `négative`. Les premières devraient être **plus petites** que les secondes.

Une façon de le vérifier est de construire une matrice 10×10 où chaque case \((i, j)\) contient la distance moyenne (ici dérivée de la similarité cosinus) entre les embeddings des images de la classe \(i\) et ceux de la classe \(j\). Sur la diagonale, on retrouve les distances intra‑classe (qui devraient être faibles si le modèle regroupe bien les images similaires). Une *heatmap* permet de visualiser rapidement la qualité de la séparation apprise.

Pour calculer cette matrice, nous commençons par construire un dictionnaire contenant les embeddings du jeu de validation, regroupés par classe.


```python
net.eval()

embeddings_by_class = {i: [] for i in range(10)} 

with torch.no_grad():
    anchor_labels = val_labels[:, 0]
    
    for idx in range(len(val_triplets)):
        label = int(anchor_labels[idx])
        
        img = torch.from_numpy(val_triplets[idx, 0].transpose(2, 0, 1) / 255.0).float()
        img = val_transforms(img)
        img = img.unsqueeze(0).to(device)
        
        embedding = net(img)
        embeddings_by_class[label].append(embedding.cpu())

embeddings_by_class = {label: torch.cat(embeddings_by_class[label], dim=0) for label in range(10)}
samples_per_class = [len(embeddings_by_class[i]) for i in range(10)]

print("Nombre d'embeddings par classe :")
for class_idx, count in enumerate(samples_per_class):
    print(f"  {label_names[class_idx]:10s} : {count}")
```

    Nombre d'embeddings par classe :
      airplane   : 124
      automobile : 113
      bird       : 125
      cat        : 119
      deer       : 127
      dog        : 138
      frog       : 119
      horse      : 123
      ship       : 130
      truck      : 132


Nous pouvons maintenant calculer la matrice de distances, puis afficher la carte de chaleur correspondante pour la rendre plus lisible.


```python
dist_matrix = np.zeros((10, 10))

for i in range(10):
    for j in range(10):
        emb_i = embeddings_by_class[i]
        emb_j = embeddings_by_class[j]
        
        emb_i_norm = F.normalize(emb_i, p=2, dim=1)
        emb_j_norm = F.normalize(emb_j, p=2, dim=1)
        
        cosine_sim = torch.mm(emb_i_norm, emb_j_norm.t())
        cosine_dist = 1 - cosine_sim
        
        dist_matrix[i, j] = cosine_dist.mean().item()

print(f"Dimension de la matrice des distances : {dist_matrix.shape}")
dist_matrix
```

<span style="background-color:#ffebcc; padding:0.1em 0.2em; font-family:'Courier New', monospace;"> > Dimension de la matrice des distances : (10, 10)</span>

    array([[0.16124831, 1.16584909, 0.75517124, 1.17487168, 1.03120434,
            1.21735156, 1.2835449 , 1.10404694, 0.74637926, 1.11012316],
           [1.16584909, 0.21340227, 1.45904374, 1.32663393, 1.41936672,
            1.32318223, 1.10714138, 1.36428154, 0.87585449, 0.65769845],
           [0.75517124, 1.45904398, 0.37313449, 0.84418499, 0.85241681,
            0.91793299, 0.9647814 , 0.9812752 , 1.11957252, 1.30334067],
           [1.17487168, 1.32663393, 0.84418499, 0.26312363, 0.74529028,
            0.36305472, 0.69436389, 0.7757206 , 1.17979074, 1.11703753],
           [1.03120446, 1.41936672, 0.85241681, 0.74529022, 0.23155765,
            0.78006458, 0.97499228, 0.70315397, 1.27855551, 1.28345919],
           [1.21735156, 1.32318223, 0.91793299, 0.36305469, 0.78006452,
            0.29617718, 0.87353909, 0.69927227, 1.25501871, 1.17946184],
           [1.2835449 , 1.10714138, 0.9647814 , 0.69436389, 0.97499228,
            0.87353909, 0.23015089, 1.19163358, 1.15588987, 1.17585552],
           [1.10404682, 1.36428154, 0.98127526, 0.7757206 , 0.70315403,
            0.69927227, 1.19163346, 0.19766319, 1.2216779 , 1.05603909],
           [0.7463792 , 0.87585443, 1.11957264, 1.17979074, 1.27855563,
            1.25501871, 1.15588987, 1.2216779 , 0.1128539 , 0.83521283],
           [1.11012328, 0.65769845, 1.30334067, 1.11703753, 1.28345907,
            1.17946196, 1.17585564, 1.05603909, 0.83521283, 0.28802815]])




```python
import seaborn as sns

plt.figure(figsize=(10, 9))

col_labels = [f"{label_names[i]}" for i in range(10)]
row_labels = [f"{label_names[i]}" for i in range(10)]

sns.heatmap(dist_matrix, 
            xticklabels=col_labels, 
            yticklabels=row_labels,
            annot=True,
            fmt='.2f',
            cmap='viridis', 
            cbar_kws={'label': 'Distance L2'})

plt.title('Matrice des distances', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

plot_filename = save_dir / "distance_matrix_heatmap.png"
plt.savefig(plot_filename)
print(f"Heatmap enregistrée dans {plot_filename}")

plt.show()
plt.close()

same_class_dists = np.diag(dist_matrix)
diff_class_dists = []
for i in range(10):
    for j in range(10):
        if i != j:
            diff_class_dists.append(dist_matrix[i, j])

print(f"\nDistances intra-classe : moyenne={np.mean(same_class_dists):.4f}, écart-type={np.std(same_class_dists):.4f}")
print(f"Distances inter-classe : moyenne={np.mean(diff_class_dists):.4f}, écart-type={np.std(diff_class_dists):.4f}")
print(f"Marge de séparation : {np.mean(diff_class_dists) - np.mean(same_class_dists):.4f}")
```

<span style="background-color:#ffebcc; padding:0.1em 0.2em; font-family:'Courier New', monospace;"> > Heatmap enregistrée dans runs/20251214_231259/distance_matrix_heatmap.png</span>

    
![png](/article_chap1_files/article_chap1_53_1.png)
    

<span style="background-color:#ffebcc; padding:0.1em 0.2em; font-family:'Courier New', monospace;">
> Distances intra-classe : moyenne=0.2367, écart-type=0.0698 <br/>
> Distances inter-classe : moyenne=1.0365, écart-type=0.2422 <br/>
> Marge de séparation : 0.7998
</span>



Plusieurs observations intéressantes ressortent de cette matrice :

**Diagonale faible** : les distances intra‑classe (sur la diagonale) sont toutes inférieures à 0,4, ce qui montre que le modèle regroupe bien les images d'une même classe. Les classes les mieux regroupées sont *ship* (0,11) et *airplane* (0,16).

**Bonne séparation globale** : la marge de séparation (différence entre la distance inter‑classe moyenne et la distance intra‑classe moyenne) est d'environ 0,80, ce qui est un bon signe.

**Confusions sémantiques attendues** : certaines paires de classes restent relativement proches, ce qui correspond à des similarités visuelles réelles :
   - *cat* et *dog* (0,36) : deux animaux à fourrure de taille similaire ;
   - *airplane* et *ship* (0,75) : véhicules aux formes allongées, avec des arrière‑plans souvent uniformes (ciel/eau) ;
   - *automobile* et *truck* (0,66) : véhicules routiers partageant des caractéristiques visuelles communes.

**Classes bien séparées** : à l'inverse, *automobile* vs *deer* (1,42) ou *bird* vs *truck* (1,20) présentent des distances élevées, cohérentes avec l'absence de ressemblance visuelle entre ces catégories.


## Analyse en Composantes principales

Il est aussi intéressant d'observer comment se répartissent nos embeddings. Comme ils vivent dans un espace à 128 dimensions, nous les projetons en 2D pour pouvoir les visualiser.

Pour cela, nous utilisons une ACP (analyse en composantes principales), appelée *Principal Component Analysis* (PCA) en anglais : une méthode linéaire de réduction de dimension.


```python
from sklearn.decomposition import PCA

all_embeddings = torch.cat([embeddings_by_class[k] for k in embeddings_by_class], dim=0)

pca_2d = PCA(n_components=2)
embeddings_2d = pca_2d.fit_transform(all_embeddings)
```

Passons maintenant à la projection. En plus des points, pour chaque classe, nous traçons une ellipse qui contient \(k\%\) des points (ici k = 50\%). Nous appelons ce paramètre `coverage`.

Commençons par écrire la fonction qui calcule les paramètres de ces ellipses. Le calcul implique une distance de Mahalanobis ainsi qu'une décomposition en valeurs/vecteurs propres ; je ne rentre pas dans le détail ici, mais je fournis le code.


```python
def compute_ellipse_parameters(embeddings, coverage):
    center = np.median(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    # Squared Mahalanobis distances and empirical quantile for coverage
    d2 = np.einsum("ij,jk,ik->i", embeddings - center, inv_cov, embeddings - center)
    threshold = np.quantile(d2, coverage)

    # Ellipse parameters from covariance eigen decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2.0 * np.sqrt(vals * threshold)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    return center, width, height, angle
```

On peut désormais calculer les paramètres des ellipses pour chaque classe.


```python
labels_array = np.concatenate([np.full(count, label) for label, count in enumerate(samples_per_class)])

coverage = 0.50

def get_ellipse_params_per_class(embeddings_2d, coverage):
    ellipse_params = {}
    for cls in label_names:
        cls_idx = label_names.index(cls)
        X = embeddings_2d[labels_array == cls_idx]
        center, width, height, angle = compute_ellipse_parameters(X, coverage)
        ellipse_params[cls] = {
            "center": center.tolist(),
            "width": float(width),
            "height": float(height),
            "angle": float(angle),
        }
    return ellipse_params

ellipse_params = get_ellipse_params_per_class(embeddings_2d, coverage)
```

Passons à la projection.


```python
import pandas as pd
from matplotlib.patches import Ellipse

def plot_embeddings_with_ellipses(
    embeddings_2d,
    ellipse_params,
    save_img_path,
):
    pca_2d_df = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'Label': labels_array,
        'Class': [label_names[int(label)] for label in labels_array]
    })

    fig, ax = plt.subplots(figsize=(12, 10))
    
    palette = sns.color_palette("tab10", n_colors=10)
    class_names = sorted(pca_2d_df['Class'].unique())
    color_map = {cls: palette[i] for i, cls in enumerate(class_names)}

    sns.scatterplot(
        data=pca_2d_df, x='PC1', y='PC2',
        hue='Class', palette=color_map,
        alpha=0.7, s=30, ax=ax
    )

    for cls in class_names:
        ep = ellipse_params[cls]
        center, w, h, angle = ep["center"], ep["width"], ep["height"], ep["angle"]
        color = color_map[cls]
        
        ellipse = Ellipse(
            xy=center, width=w, height=h, angle=angle,
            facecolor=(*color, 0.12), edgecolor=color, linewidth=2
        )
        ax.add_patch(ellipse)

    ax.set_xlabel('CP1')
    ax.set_ylabel('CP2')
    ax.set_title('Projection ACP 2D des embeddings par classe')
    ax.legend(title='Classes', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_img_path, dpi=150, bbox_inches='tight')
    plt.show()


plot_embeddings_with_ellipses(
    embeddings_2d,
    ellipse_params,
    save_img_path = save_dir / "embeddings_2d.png"
)
```


    
![png](/article_chap1_files/article_chap1_62_0.png)
    


Le résultat dépend de la `seed` choisie. À moins d'exécuter exactement le même code (et avec la même `seed`), vous obtiendrez forcément une projection différente. Néanmoins, on retrouve généralement quelques tendances :
- *ship* et *airplane* ont les ellipses de plus petite aire, ce qui confirme l'analyse de la matrice de distances
- les points *cat* et *dog* sont proches, comme attendu
- autre confirmation de la matrice de distances : *automobile* et *deer* sont bien séparés, tout comme *bird* et *truck*.

Avant de terminer cet article, nous normalisons les coordonnées projetées entre 0 et 1, puis nous redessinons le plan. Cela nous servira ensuite à comparer plus facilement les valeurs entre expériences : paramètres des ellipses pour certaines classes, ainsi que la moyenne et la médiane de leurs aires.

Dans la suite, nous introduirons la *KoLeo loss*, qui aura un impact direct sur la répartition des embeddings et donc sur la taille des ellipses.


```python
embeddings_2d = (embeddings_2d - embeddings_2d.min(axis=0)) / (embeddings_2d.max(axis=0) - embeddings_2d.min(axis=0))
```


```python
ellipse_params = get_ellipse_params_per_class(embeddings_2d, coverage)
plot_embeddings_with_ellipses(
    embeddings_2d=embeddings_2d,
    ellipse_params=ellipse_params,
    save_img_path = save_dir / "embeddings_2d_normalized.png"
)
```


    
![png](/article_chap1_files/article_chap1_66_0.png)
    



```python
mean_area = 0
for ep_dict in ellipse_params.values():
    area = np.pi * ep_dict["width"] * ep_dict["height"]
    ep_dict["area"] = area
    mean_area += area
```


```python
for k, v in ellipse_params.items():
    if k in ["ship", "cat", "dog", "horse"]:
        print(f"Aire de l'ellipse de {k} = {v['area']:.4f}")
```

<span style="background-color:#ffebcc; padding:0.1em 0.2em; font-family:'Courier New', monospace;"> > Aire de l'ellipse de cat = 0.0690</span>
<span style="background-color:#ffebcc; padding:0.1em 0.2em; font-family:'Courier New', monospace;"> > Aire de l'ellipse de dog = 0.0546</span>
<span style="background-color:#ffebcc; padding:0.1em 0.2em; font-family:'Courier New', monospace;"> > Aire de l'ellipse de horse = 0.0450</span>
<span style="background-color:#ffebcc; padding:0.1em 0.2em; font-family:'Courier New', monospace;"> > Aire de l'ellipse de ship = 0.0083</span>


# Conclusion

Dans cet article, nous avons analysé les embeddings obtenus après l'entraînement d'un réseau siamois avec une *triplet loss* sur CIFAR‑10. Les résultats montrent une bonne séparation des classes dans l'espace des embeddings, et les confusions observées (*cat*/*dog*, *airplane*/*ship*) correspondent à des similarités visuelles réelles. La matrice de distances et la projection par ACP (PCA) avec les ellipses de confiance permettent de visualiser et de quantifier la compacité de chaque groupe.

Dans la suite de cette série, nous introduirons la *KoLeo loss* pour encourager une répartition plus uniforme des embeddings, et nous étudierons l'impact de l'accumulation de *gradient* sur cette régularisation dépendante du batch.

