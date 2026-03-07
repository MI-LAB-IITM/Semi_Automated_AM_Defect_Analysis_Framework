# SMILE
**Sampling via Maximin Latin Hypercube Sampling from Embeddings**

SMILE is a **core-set based subset selection algorithm** designed for **batch active learning and data-efficient machine learning**.  
The method selects **diverse and representative samples** from a dataset by combining **clustering-based coverage** with **Latin Hypercube Sampling (LHS) and a maximin diversity criterion** in a 2D embedding space.

The goal is to **maximize feature-space coverage while minimizing redundancy**, which is particularly useful when labeling data is expensive.

---

## Method

SMILE operates on **2D embeddings** (e.g., generated using **t-SNE** or **UMAP**) of dataset features.

The algorithm follows four main steps:

1. **Embedding Projection**  
   Unlabeled samples are projected into a **2D embedding space**, where local neighborhood relationships between samples are preserved.

2. **Clustering**  
   The embedding space is partitioned using **K-means clustering**, grouping samples that represent similar regions of the feature space.

3. **Cluster Ranking**  
   Each cluster is ranked based on a **spread metric**, defined as the Euclidean norm of the standard deviation along both embedding dimensions:

   spread = sqrt(σx² + σy²)
   Clusters with higher spread represent **greater variability** and are prioritized during active learning rounds.

4. **Subset Selection (LHS + Maximin)**  
   From each cluster, **k representative samples** are selected using:

   - **Strict Latin Hypercube Sampling (LHS)**  
     Ensures one sample per row and column in a k × k grid over the normalized embedding space.

   - **Relaxed Maximin Sampling**  
     If strict LHS is infeasible due to insufficient bin coverage, the algorithm selects samples that **maximize pairwise spacing** using a maximin criterion.

This approach ensures **balanced spatial coverage and high diversity** among selected samples.

## Repository Structure

```
SMILE/
│
├── SMILE.py               # Algorithm implementation
├── SMILE_notebook.ipynb   # Example notebook demonstrating SMILE
├── readme.md              
```
