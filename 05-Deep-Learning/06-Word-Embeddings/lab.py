import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


def train_word2vec(sentences, vector_size=50, window=3, sg=1, epochs=200, seed=42):
    """Train Word2Vec on a custom corpus.

    Args:
        sentences: List of lists of tokenized words.
        vector_size: Embedding dimensionality.
        window: Context window size.
        sg: 1 for Skip-Gram, 0 for CBOW.
        epochs: Number of training iterations.
        seed: Random seed for reproducibility.

    Returns:
        Trained Word2Vec model.
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=1,
        sg=sg,
        epochs=epochs,
        seed=seed,
    )
    return model


def demo_similarity(model, word_pairs):
    """Print cosine similarity for a list of word pairs."""
    print(f"\n{'Word1':12s} {'Word2':12s} {'Similarity':>10s}")
    print("-" * 36)
    for w1, w2 in word_pairs:
        if w1 in model.wv and w2 in model.wv:
            sim = model.wv.similarity(w1, w2)
            print(f"{w1:12s} {w2:12s} {sim:>10.4f}")
        else:
            print(f"{w1:12s} {w2:12s} {'N/A':>10s}")


def demo_analogy(model, positive, negative, topn=5):
    """Demonstrate word analogy: positive[0] - negative[0] + positive[1] = ?"""
    query = f"{positive[0]} - {negative[0]} + {positive[1]}"
    print(f"\n  Analogy: {query} = ?")
    results = model.wv.most_similar(positive=positive, negative=negative, topn=topn)
    for word, score in results:
        print(f"    {word:12s} {score:.4f}")
    return results


def visualize_pca(model, words=None, title="Word Embeddings (PCA to 2D)"):
    """Visualize word embeddings using PCA."""
    if words is None:
        words = list(model.wv.index_to_key)
    vectors = np.array([model.wv[w] for w in words])

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)

    _, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=80)
    for i, word in enumerate(words):
        ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=9)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def visualize_tsne(model, words=None, color_map=None, title="Word Embeddings (t-SNE Visualization)"):
    """Visualize word embeddings using t-SNE with color-coded clusters."""
    if words is None:
        words = list(model.wv.index_to_key)
    vectors = np.array([model.wv[w] for w in words])

    # PCA to reduce to 20 dimensions first (t-SNE performs better)
    pca = PCA(n_components=min(20, len(words) - 1), random_state=42)
    vectors_pca = pca.fit_transform(vectors)

    # t-SNE to 2D
    n_components = min(2, len(words) - 1)
    tsne = TSNE(n_components=n_components, random_state=42,
                perplexity=min(5, len(words) - 1))
    coords = tsne.fit_transform(vectors_pca)

    # Color coding
    if color_map is None:
        color_map = {
            'royal': 'purple',
            'person': 'blue',
            'place': 'green',
            'child': 'orange',
            'country': 'red',
            'other': 'gray',
        }

    colors = []
    for w in words:
        if w in ('king', 'queen', 'prince', 'princess', 'throne'):
            colors.append('royal')
        elif w in ('man', 'woman'):
            colors.append('person')
        elif w in ('boy', 'girl'):
            colors.append('child')
        elif w in ('paris', 'london', 'berlin', 'rome', 'france', 'england', 'germany', 'italy',
                   'capital', 'madrid', 'spain', 'lisbon', 'portugal'):
            colors.append('country')
        else:
            colors.append('other')

    color_values = [color_map[c] for c in colors]

    _, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=color_values, alpha=0.7, s=100)
    for i, word in enumerate(words):
        ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=9)

    # Add legend
    legend_elements = []
    for label, color in color_map.items():
        if label in colors:
            legend_elements.append(plt.scatter([], [], c=color, label=label, s=50))
    ax.legend(handles=legend_elements, loc='best')

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def run_embeddings_lab():
    print("=" * 60)
    print("WORD EMBEDDINGS LAB: Word2Vec from Scratch")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Create Custom Corpus
    # ---------------------------------------------------------
    print("\n[1] Building custom corpus...")
    sentences = [
        # Royal family
        ["the", "king", "sat", "on", "the", "throne"],
        ["the", "queen", "sat", "on", "the", "throne"],
        ["the", "prince", "fought", "the", "dragon"],
        ["the", "princess", "fought", "the", "dragon"],
        ["the", "king", "and", "queen", "ruled", "the", "kingdom"],
        ["the", "prince", "and", "princess", "lived", "in", "the", "castle"],
        ["the", "throne", "belonged", "to", "the", "king"],

        # People
        ["the", "man", "walked", "to", "the", "market"],
        ["the", "woman", "walked", "to", "the", "market"],
        ["the", "man", "bought", "bread", "at", "the", "market"],
        ["the", "woman", "bought", "fruit", "at", "the", "market"],

        # Children
        ["the", "boy", "played", "in", "the", "park"],
        ["the", "girl", "played", "in", "the", "park"],
        ["the", "boy", "ran", "in", "the", "garden"],
        ["the", "girl", "ran", "in", "the", "garden"],

        # Capitals and countries
        ["paris", "is", "the", "capital", "of", "france"],
        ["london", "is", "the", "capital", "of", "england"],
        ["berlin", "is", "the", "capital", "of", "germany"],
        ["rome", "is", "the", "capital", "of", "italy"],
        ["madrid", "is", "the", "capital", "of", "spain"],
        ["lisbon", "is", "the", "capital", "of", "portugal"],

        # Geography sentences
        ["france", "is", "a", "country", "in", "europe"],
        ["germany", "is", "a", "country", "in", "europe"],
        ["spain", "is", "a", "country", "in", "europe"],
        ["the", "eiffel", "tower", "is", "in", "paris"],
        ["the", "brandenburg", "gate", "is", "in", "berlin"],
        ["the", "colosseum", "is", "in", "rome"],
    ]

    print(f"    Number of sentences: {len(sentences)}")
    print(f"    Total tokens: {sum(len(s) for s in sentences)}")

    # ---------------------------------------------------------
    # 2. Train Word2Vec (Skip-Gram)
    # ---------------------------------------------------------
    print("\n[2] Training Word2Vec (Skip-Gram, 50D embeddings)...")
    model_sg = train_word2vec(sentences, vector_size=50, sg=1)
    print(f"    Vocabulary size: {len(model_sg.wv)}")
    print(f"    Embedding dimension: {model_sg.wv.vector_size}")

    # ---------------------------------------------------------
    # 3. Train Word2Vec (CBOW)
    # ---------------------------------------------------------
    print("\n[3] Training Word2Vec (CBOW, 50D embeddings)...")
    model_cbow = train_word2vec(sentences, vector_size=50, sg=0)
    print(f"    Vocabulary size: {len(model_cbow.wv)}")
    print(f"    Embedding dimension: {model_cbow.wv.vector_size}")

    # ---------------------------------------------------------
    # 4. Semantic Similarity (Skip-Gram)
    # ---------------------------------------------------------
    print("\n[4] Semantic Similarity (Skip-Gram model):")
    pairs = [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"),
        ("paris", "france"), ("king", "boy"), ("paris", "london"),
        ("king", "man"), ("queen", "woman"),
    ]
    demo_similarity(model_sg, pairs)

    # ---------------------------------------------------------
    # 5. Analogies
    # ---------------------------------------------------------
    print("\n[5] Analogies (Skip-Gram model):")
    print("-" * 40)

    # king - man + woman = ?
    demo_analogy(model_sg, positive=["king", "woman"], negative=["man"], topn=5)

    # paris - france + germany = ?
    demo_analogy(model_sg, positive=["paris", "germany"], negative=["france"], topn=5)

    # boy - man + woman = ?
    demo_analogy(model_sg, positive=["boy", "woman"], negative=["man"], topn=5)

    # ---------------------------------------------------------
    # 6. Compare Skip-Gram vs CBOW on the same analogy
    # ---------------------------------------------------------
    print("\n[6] Skip-Gram vs CBOW comparison:")
    print("-" * 40)
    sg_result = model_sg.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
    cbow_result = model_cbow.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
    print(f"    Skip-Gram: king - man + woman ~= {sg_result[0][0]} ({sg_result[0][1]:.4f})")
    print(f"    CBOW:      king - man + woman ~= {cbow_result[0][0]} ({cbow_result[0][1]:.4f})")

    # ---------------------------------------------------------
    # 7. Nearest Neighbors
    # ---------------------------------------------------------
    print("\n[7] Nearest neighbors for key words:")
    print("-" * 40)
    for word in ["king", "paris", "man"]:
        if word in model_sg.wv:
            neighbors = model_sg.wv.most_similar(word, topn=5)
            neighbor_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbors)
            print(f"    {word}: {neighbor_str}")

    # ---------------------------------------------------------
    # 8. PCA Visualization
    # ---------------------------------------------------------
    print("\n[8] Generating visualizations...")

    # All words with color coding
    colors = []
    all_words = list(model_sg.wv.index_to_key)
    for w in all_words:
        if w in ("king", "queen", "prince", "princess", "throne", "castle", "kingdom"):
            colors.append("royal")
        elif w in ("man", "woman"):
            colors.append("person")
        elif w in ("boy", "girl", "played", "ran"):
            colors.append("child")
        elif w in ("paris", "london", "berlin", "rome", "madrid", "lisbon",
                   "france", "england", "germany", "italy", "spain", "portugal",
                   "capital", "europe"):
            colors.append("country")
        else:
            colors.append("other")

    # PCA
    vectors = np.array([model_sg.wv[w] for w in all_words])
    pca_coords = PCA(n_components=2, random_state=42).fit_transform(vectors)

    color_map_val = {
        'royal': 'purple', 'person': 'blue', 'child': 'orange',
        'country': 'green', 'other': 'gray'
    }
    color_values = [color_map_val[c] for c in colors]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    ax1.scatter(pca_coords[:, 0], pca_coords[:, 1], c=color_values, alpha=0.7, s=80)
    for i, word in enumerate(all_words):
        ax1.annotate(word, (pca_coords[i, 0], pca_coords[i, 1]), fontsize=8)
    ax1.set_title("Word Embeddings (PCA to 2D)")
    ax1.grid(True, alpha=0.3)

    # t-SNE (with PCA preprocessing for stability)
    pca_50 = PCA(n_components=min(20, len(all_words) - 1), random_state=42)
    vectors_pca = pca_50.fit_transform(vectors)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(all_words) - 1))
    tsne_coords = tsne.fit_transform(vectors_pca)

    ax2.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=color_values, alpha=0.7, s=80)
    for i, word in enumerate(all_words):
        ax2.annotate(word, (tsne_coords[i, 0], tsne_coords[i, 1]), fontsize=8)
    ax2.set_title("Word Embeddings (t-SNE to 2D)")
    ax2.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        plt.scatter([], [], c='purple', label='Royal', s=50),
        plt.scatter([], [], c='blue', label='Person', s=50),
        plt.scatter([], [], c='orange', label='Child', s=50),
        plt.scatter([], [], c='green', label='Country/City', s=50),
        plt.scatter([], [], c='gray', label='Other', s=50),
    ]
    ax2.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig('word_embeddings_visualization.png', dpi=150)
    print("    Saved: word_embeddings_visualization.png")

    # ---------------------------------------------------------
    # 9. Embedding Math Summary
    # ---------------------------------------------------------
    print("\n[9] Embedding Arithmetic Summary:")
    print("-" * 40)

    analogies = [
        (["king", "woman"], ["man"], "king - man + woman"),
        (["paris", "germany"], ["france"], "paris - france + germany"),
        (["madrid", "france"], ["spain"], "madrid - spain + france"),
        (["boy", "woman"], ["man"], "boy - man + woman"),
        (["london", "france"], ["england"], "london - england + france"),
    ]

    for pos, neg, desc in analogies:
        try:
            result = model_sg.wv.most_similar(positive=pos, negative=neg, topn=1)
            print(f"    {desc:35s} ~= {result[0][0]:10s} ({result[0][1]:.4f})")
        except KeyError as e:
            print(f"    {desc:35s} ~= ERROR: {e}")

    print("\n" + "=" * 60)
    print("Word Embeddings Lab Complete")
    print("=" * 60)


if __name__ == "__main__":
    run_embeddings_lab()
