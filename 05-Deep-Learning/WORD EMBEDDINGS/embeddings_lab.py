import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def run_embeddings_lab():
 # ---------------------------------------------------------
 # 1. Train Word2Vec on Custom Text (No internet needed)
 # ---------------------------------------------------------
 print("=" * 50)
 print("PART 1: Train Word2Vec from Scratch")
 print("=" * 50)

 from gensim.models import Word2Vec

 # Simulated corpus
 sentences = [
 ["the", "king", "sat", "on", "the", "throne"],
 ["the", "queen", "sat", "on", "the", "throne"],
 ["the", "prince", "fought", "the", "dragon"],
 ["the", "princess", "fought", "the", "dragon"],
 ["the", "man", "walked", "to", "the", "market"],
 ["the", "woman", "walked", "to", "the", "market"],
 ["the", "boy", "played", "in", "the", "park"],
 ["the", "girl", "played", "in", "the", "park"],
 ["paris", "is", "the", "capital", "of", "france"],
 ["london", "is", "the", "capital", "of", "england"],
 ["berlin", "is", "the", "capital", "of", "germany"],
 ["rome", "is", "the", "capital", "of", "italy"],
 ]

 model = Word2Vec(sentences, vector_size=50, window=3, min_count=1,
 workers=1, sg=1, epochs=200, seed=42)

 print(f"Vocabulary size: {len(model.wv)}")
 print(f"Embedding dimension: {model.wv.vector_size}")

 # ---------------------------------------------------------
 # 2. Similarity Queries
 # ---------------------------------------------------------
 print("\n Similarity Queries:")
 pairs = [("king", "queen"), ("man", "woman"), ("boy", "girl"),
 ("paris", "france"), ("king", "boy")]
 for w1, w2 in pairs:
 sim = model.wv.similarity(w1, w2)
 print(f" {w1:10s} ↔ {w2:10s}: {sim:.3f}")

 # ---------------------------------------------------------
 # 3. Analogy: king - man + woman ≈ queen
 # ---------------------------------------------------------
 print("\n Analogy Test: king - man + woman = ?")
 result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
 for word, score in result:
 print(f" {word}: {score:.3f}")

 # ---------------------------------------------------------
 # 4. Visualization (2D)
 # ---------------------------------------------------------
 words = list(model.wv.index_to_key)
 vectors = np.array([model.wv[w] for w in words])

 # PCA to 50D first (for t-SNE)
 pca = PCA(n_components=min(20, len(words) - 1))
 vectors_pca = pca.fit_transform(vectors)

 # t-SNE to 2D
 n_components = min(2, len(words) - 1)
 tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(5, len(words) - 1))
 coords = tsne.fit_transform(vectors_pca)

 plt.figure(figsize=(10, 7))
 colors = []
 color_map = {
 'royal': 'purple', 'person': 'blue', 'place': 'green',
 'child': 'orange', 'country': 'red'
 }
 for w in words:
 if w in ('king', 'queen', 'prince', 'princess', 'throne'):
 colors.append('purple')
 elif w in ('man', 'woman'):
 colors.append('blue')
 elif w in ('boy', 'girl'):
 colors.append('orange')
 elif w in ('paris', 'london', 'berlin', 'rome', 'france', 'england', 'germany', 'italy'):
 colors.append('green')
 else:
 colors.append('gray')

 plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.7, s=100)
 for i, word in enumerate(words):
 plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=9)
 plt.title('Word Embeddings (t-SNE Visualization)')
 plt.grid(True, alpha=0.3)
 plt.tight_layout()
 plt.show()

 # ---------------------------------------------------------
 # 5. Embedding Math
 # ---------------------------------------------------------
 print("\n Embedding Arithmetic:")
 print(f" king - man + woman ≈ {model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)[0][0]}")
 print(f" paris - france + germany ≈ {model.wv.most_similar(positive=['paris', 'germany'], negative=['france'], topn=1)[0][0]}")

if __name__ == "__main__":
 run_embeddings_lab()
