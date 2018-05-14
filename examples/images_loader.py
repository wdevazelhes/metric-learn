from itertools import chain
import numpy as np
import os

from functools import reduce

from metric_learn import LMNN, ITML, ITMLTransformer, LFDA
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.datasets.lfw import _load_imgs
from sklearn.datasets import get_data_home
from six import string_types
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import LargeMarginNearestNeighbors
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from skimage.feature import daisy, hog
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from pathlib import Path
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt



class FacesInTheWildLoader(BaseEstimator, TransformerMixin):

    def __init__(self, root_path='', color=True, resize=1.,
                 crop_slice=(slice(70, 195), slice(78, 172))):
        self.root_path = root_path
        self.crop_slice = crop_slice
        self.color = color
        self.resize = resize

    def transform(self, paths):
        paths = [os.path.join(self.root_path, path) for path in paths]
        return _load_imgs(paths, self.crop_slice, self.color, self.resize)

class DaisyFeatureExtractor(BaseEstimator, TransformerMixin):

    def transform(self, images):
        shape = (len(images), np.prod(hog(images[0] / 255.).shape))
        features = np.zeros(shape)
        for i, image in enumerate(images):
            features[i, ...] = hog(image / 255.).ravel()
        return features

    def fit(self, X):
        pass

class FlattenTransformer(BaseEstimator, TransformerMixin):

    def transform(self, images):
        len_images = len(images)
        return images.reshape((len_images, -1))

    def fit(self, X):
        pass

lfw_root_path = os.path.join(get_data_home(), "lfw_home", "lfw_funneled")
loader = FacesInTheWildLoader(root_path=lfw_root_path, color=False,
                              resize=0.5)

sample_images = loader.transform([
    "Zinedine_Zidane/Zinedine_Zidane_0001.jpg",
    "Zinedine_Zidane/Zinedine_Zidane_0002.jpg",
    "Andre_Agassi/Andre_Agassi_0001.jpg",
])

print(sample_images[0].shape)
plt.imshow(sample_images[0])
plt.show()


# feature_extractor = make_pipeline(loader, FlattenTransformer())
feature_extractor = make_pipeline(loader, DaisyFeatureExtractor())

features = feature_extractor.transform([
    "Zinedine_Zidane/Zinedine_Zidane_0001.jpg",
    "Zinedine_Zidane/Zinedine_Zidane_0002.jpg",
    "Andre_Agassi/Andre_Agassi_0001.jpg"])

# nca = LMNN()
nca = NeighborhoodComponentsAnalysis(n_features_out=5, max_iter=100,
    verbose=4)

absolute_paths = list(Path(lfw_root_path).glob("**/*.jpg"))
paths = [path.relative_to(lfw_root_path) for path in absolute_paths]
X = list(map(str, paths))
y = [file.split(os.path.sep, 1)[0] for file in X]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True,
                                             train_size=0.3, random_state=0)
y_train = LabelEncoder().fit_transform(y_train)

nca_train = nca.fit_transform(feature_extractor.transform(X_train), y_train)
# nca_train = nca.fit_transform(feature_extractor.transform(X_train), y_train)

pca = PCA(n_components=2)
pca_train = pca.fit_transform(nca_train)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(pca_train[:, 0], pca_train[:, 1], c=y_train, marker='o',
           label='train')
ax.legend(loc='best')
plt.show()