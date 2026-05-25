# Simulateur d'Érosion Hydraulique et Thermique

Ce projet est un simulateur interactif de génération procédurale de terrains et d'érosion, exécuté en temps réel sur carte graphique (GPU). Il implémente les modèles d'érosion hydraulique de Mei et al. (2007) et d'érosion thermique de Jákó et Szirmay-Kalos (2011) en utilisant l'API OpenGL et les Compute Shaders.

## Fonctionnalités

- **Érosion Hydraulique Temps Réel :** Simulation de la dynamique des fluides pour creuser des lits de rivières et créer des dépôts deltaïques.
- **Érosion Thermique :** Simulation des éboulements gravitationnels pour transformer les falaises abruptes en cônes d'éboulis naturels.
- **Accélération Matérielle :** Pipeline entièrement déporté sur le GPU via Compute Shaders (Ping-Pong Buffering) atteignant plus de 900 FPS.
- **Interface Interactive :** Contrôle des paramètres physiques (pluie, capacité de transport, évaporation) en temps réel via Dear ImGui.

## Prérequis

- **OpenGL 4.3** minimum (nécessaire pour les Compute Shaders).
- **CMake** (version 3.10+).
- Un compilateur **C++17** (GCC, Clang ou MSVC).
- Bibliothèques externes : `GLFW`, `GLEW` (ou GLAD), `GLM`, et `Dear ImGui`.

## Compilation et Exécution

### Sous Linux

Assurez-vous d'avoir les dépendances de développement installées :

```bash
gcc c++, cmake, glfw, glew, glm
```

Pour lancer le projet, mettez vous à la racine (/hydraulic_erosion) et faites les commandes suivantes :

```bash
mkdir build
cd build
make -j
./hydraulic_erosion
```

En une commande :

```bash
mkdir build && cd build && make -j && ./hydraulic_erosion
```

## Architecture du Projet

Le projet sépare strictement la logique d'application (C++) de la simulation physique (GLSL).

### Fichiers Sources (hydraulic_erosion.cpp)

- hydraulic_erosion.cpp : Point d'entrée, initialisation du contexte OpenGL, de la fenêtre GLFW et de la boucle principale, c'est le cœur du programme côté CPU

- hydraulic_erosionCPU.cpp : contient notre version **FULL** CPU de l'implémentation de notre projet

### Pipeline de Shaders (/shaders)

La simulation est découpée en 8 passes de Compute Shaders distinctes :

- water_increment.glsl : Ajoute la pluie et les sources d'eau locales.

- flux_compute.glsl : Calcule la pression hydrostatique et le débit d'eau vers les voisins (Virtual Pipes).

- water_update.glsl : Met à jour la hauteur d'eau et infère le vecteur vitesse tridimensionnel du fluide.

- erosion_deposition.glsl : Gère l'arrachement de la roche et le dépôt des sédiments selon la capacité de transport.

- sediment_transport.glsl : Déplace les sédiments avec le courant (Advection semi-Lagrangienne).

- evaporation.glsl : Évapore l'eau et applique les conditions aux limites ouvertes.

- thermal_flux.glsl : Calcule les instabilités terrestres et les éboulements (Érosion thermique).

- thermal_apply.glsl : Applique les modifications topologiques finales en toute sécurité (Gathering).

### Rendu Visuel :

- terrain_vertex.glsl : Applique le Vertex Displacement (déforme la grille 3D en fonction de la texture de hauteur calculée).

- terrain_fragment.glsl : Calcule les normales à la volée et applique l'ombrage procédural (roche, herbe, eau, sédiments).

### Heightmaps (carte de hauteur) (/heightmaps)

Les différentes cartes de hauteurs préchargés dans notre simulation sont dans ce dossier

- aples_francaise.png : représentant une partie des Alpes Françaises

- grand_canyon_heightmap.png : permettant de visualiser le grand canyon

- heightmap1.png : chaîne de montagne lambda

- lac_titicaca.png : une heightmap permettant de visualiser un sol plat (lac)

- nepal_mountain_range.png : représentant une chaîne de montagne népalaises

- reunion.png : heightmaps représentant l'ile de la Réunion

- thermal_height.png : heightmap permettant de visualiser facilement l'effet de l'érosion thermale

## Auteurs

Projet de TER (Master 1 IMAGINE - Université de Montpellier) réalisé par :

- César CHARBEY

- Thomas BARASCUD

- Jérémy NICOLAS

- Lucas PAULO

### Encadré par Nicolas Lutz.
