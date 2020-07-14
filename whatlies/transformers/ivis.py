from whatlies import EmbeddingSet
from whatlies.transformers.common import embset_to_X, new_embedding_dict

from ivis import Ivis as IVIS
import numpy as np


class Ivis:
    """
    This transformer scales all the vectors in an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]
    by means of Ivis algorithm. We're using the implementation found in
    [beringresearch/ivis](https://github.com/beringresearch/ivis)

    Arguments:
        n_components: the number of compoments to create/add
        kwargs: keyword arguments passed to the Ivis implementation

    Usage:

    ```python
    from whatlies.language import GensimLanguage
    from whatlies.transformers import Ivis

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = GensimLanguage("wordvectors.kv")
    emb = lang[words]

    emb.transform(Ivis(3)).plot_interactive_matrix('ivis_0', 'ivis_1', 'ivis_2')
    ```
    """

    def __init__(self, n_components=2, **kwargs):
        self.is_fitted = False
        self.n_components = n_components
        self.kwargs = kwargs

    def __call__(self, embset):
        if not self.is_fitted:
            self.fill_arguments(embset)
            self.tfm = IVIS(embedding_dims=self.n_components, **self.kwargs)
            self.fit(embset)
        return self.transform(embset)

    def fill_arguments(self, embset):
        """
        Filling arguments for Ivis Fit function:
            k: The number of neighbours to retrieve for each point.
               Must be less than one minus the number of rows in the dataset.

            batch_size: The size of mini-batches used during gradient
                        descent while training the neural network. Must be less than or
                        equal to the num_rows in the dataset.

            verbose: Controls the volume of logging output the model
                     produces when training.
        """
        size = len(embset.embeddings)
        if 'k' not in self.kwargs:
            self.kwargs['k'] = min(size-2, 150)
        if 'batch_size' not in self.kwargs:
            self.kwargs['batch_size'] = min(size, 128)
        self.kwargs['verbose'] = 0

    def fit(self, embset):
        names, X = embset_to_X(embset=embset)
        self.tfm.fit(X)
        self.is_fitted = True

    def transform(self, embset):
        names, X = embset_to_X(embset=embset)
        new_vecs = self.tfm.fit_transform(X)
        names_out = names + [f"ivis_{i}" for i in range(self.n_components)]
        vectors_out = np.concatenate([new_vecs, np.eye(self.n_components)])
        new_dict = new_embedding_dict(names_out, vectors_out, embset)
        return EmbeddingSet(new_dict, name=f"{embset.name}.ivis_{self.n_components}()")
