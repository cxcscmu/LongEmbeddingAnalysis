import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pickle
import sys

embeddings_path = sys.argv[1]
out_path = sys.argv[2]

def compute_cosine_similarity(arr1, arr2):
    dot_product = np.dot(arr1, arr2)
    magnitude_arr1 = np.linalg.norm(arr1)
    magnitude_arr2 = np.linalg.norm(arr2)
    cosine_similarity = dot_product / (magnitude_arr1 * magnitude_arr2)
    return cosine_similarity

full_embeddings = f"{embeddings_path}/full/embeddings.corpus.rank.0.0-30000"

with open(full_embeddings, 'rb') as h:
    full_vec = {}
    data = pickle.load(h)
    for vec, did in zip (data[0], data[1]):
        full_vec[did] = vec

vecs = []
for i in range(10):
    start_embeddings = f"{embeddings_path}/{i}/embeddings.corpus.rank.0.0-23904"
    with open(start_embeddings, 'rb') as h:
        start_vec = {}
        data = pickle.load(h)
        for vec, did in zip (data[0], data[1]):
            start_vec[did] = vec
        vecs.append(start_vec)


dists = []

for vec in vecs:
    start_dist = []
    for did in vec:
        start_dist.append(compute_cosine_similarity(full_vec[did], vec[did]))
    
    dists.append(start_dist)

data = dists

# Plot boxplots
plt.figure(figsize=(10, 6))
boxplot_dict = sns.violinplot(data, color='#75B0DE', bw_adjust=0.9)
plt.xlabel('Distribution')
plt.ylabel('cosine similarity')
plt.grid(True)

plt.ylim(0,1)
plt.xticks(ticks=range(0, len(data)), labels=range(1, len(data) + 1))

plt.savefig(f'{out_path}/violin_plot.pdf', bbox_inches='tight')

