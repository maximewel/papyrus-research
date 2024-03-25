# Dictionary
* Ductus: Ensemble des caractéristiques importantes pour tracer les différents traits composant une lettre. ex: _Ductus de gauche à droite_, _ductus d'une langue occidentale_.

# Concepts
Online / offline for writing: 
* Offline   : Analysis of a piece of written paper by taking its photography, f.ex. via a CNN or MLP. Concepte te __manuscrit__.
* Online    : Analysis of a piece of written paper via attributes taken when it was written. The kinematic of the penship (Speed, Ductus, pressure) is taken into account. Can be done thanks to modern written techniques that often rely on a mix of technology applied to standard scribe-related feels (Ex: Tablets). Concept of __tapuscrit__.

* Fully connected layers (FCNN) are good at finding good vectorisation of images


# Ideas
Can we use hardware to combine the feels of writing on paper with the usability of digital note-typing ?

Multiple steps to construt online model from offline images:
- Correction of perspective / distortions
- Binarization
- Cleaning artifacts

* Some parts of the reconstruction of the in-line writing is hard to reconstruct, especially across languages and habits/regional specificities.

Flow: Online image --reconstruct--> Inline data
The inverse flow is called Rasterisation and is often more feasible: Online data -> Creation of offline images

Example of online data:
- 2D coordinates x,y
- Speed
- Acceleration
- Pressure

## Evaluation metrics
* Compare to online ground truth
* generate offline images to compare
* Real humans comparing generated images
* Use online recognition as a better reconstructed signal is easier to read

Standardized methods
- RMSE expected / estimated trajectory
- DTW

## Sota
Two families
- Global optimization of graph walk
- Local resolution of ambiguities

First approach work on all features on gray levels, meanwhile second approach assumes that humans write correctly and skeletize the words first

# Challenges
- Be general
- Collect data, very expensive / not widespread
- Evaluate quality of reconstruction

# Datasets
_"(comme dans les bases IRONOFF[VG+99] et CROHME2023[Xie+23]"_

- **Isolated letters**: 
    - **UNIPEN** - First online database to be captured. Contains some words (isolated letters/signs), contain online data. Used: 1A, 1B, 1C
- **Words, sentences**: 
    - **IRONOFF** contains 32K letters, 5k words, in cursive. Onine and offline.
        - *"Cependant, le désalignement inévitable et observé dans cet appariement freine tout apprentissage supervisé."* <-- ?
    - **IAM-OnDB**, a large public DB with online data. 13k lines, 86k words obtains on white board. 
- **Mathematical expressions**
    - **CROHME**: Online, Offline mathematical equations. 12k formulas, 101 symbols.

# Thesis

- Uses DTW seg. A DTW that doesn't compute the distance between two ponts, but rather the segment representation.

SET/SORT: Dual transformers, each one with a role.
SET: Create a description vector for each trait
SORT: Predict the trait orders and pencil up

Subtrait: Set of points forming a line, with a start and an end and non-looping.

Oracle: Order of subtraits built back from online signal, where traits are associated to their corresponding online signal. Used as a ground truth.

problem: Transformer has quadratic computing based on input vector's length. Need to select appropriate widow for segment. Solution: Simiarily to max pooling, image is segmented and processed by segment.



# Questions ?

- SETSORT reconstructs ONLINE signal from OFFLINE data. Need to compartmentalize to obtain lines from document ? How do dataset work when captured via sensors with line separation ?
- Artifacts / source code available for reproductibility ?
- Uses reconstructed online images ? Tests on real-life data ? Does pre-processing still work on flawed data (jonction points) ?