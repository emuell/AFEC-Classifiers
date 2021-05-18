# AFEC Classifiers

This repository holds the sample-train-packs which are used to create [AFEC](https://github.com/emuell/afec)'s default category and classification models. 

Please note that *only* the extracted audio-features for the machine learning models and folder structure is included here. The actual audio-files of the classification packs partly got lost and also can't be uploaded because of unclear copyright issues.

But you can use the extracted features (`afec-ll.csv`) to test various classification models, and can use the given folder structure for your own classification experiments.

## Scripts

The `./Scripts` folder contains various python scripts to experiment with Keras / Tensorflow classification models.<br/> 
See individual `train_XXX.py` files for details and pip dependencies. Please note: AFEC does not use those Keras models - they are *only* used for the sake of experimentation. 

## Classification Packs

* `OneShot-vs-Loops`: 
__Class__ in AFEC: splits sounds into either _Loop_ or _OneShot_ sounds.

* `OneShot-Categories`: 
__Categories__ in AFEC: detects categories for all sounds which got classified as _OneShots_.


## Creating new or updating existing models

The AFEC crawler executable uses two bundled [shark models](https://github.com/emuell/AFEC/tree/master/Source/Classification/Export/ClassificationModel.h) to evaluate the classifiers. The shark models get built from the sample-packs in this repository.

To update the existing or built new models, you can use the `Create%SAMPLE_PACK_NAME%Model.[bat|sh]` scripts in the main AFEC main repository at [Scripts/ModelCreator](https://github.com/emuell/AFEC/tree/master/Scripts/ModelCreator).

They do the following for each sample pack:

```bash
# Create or update low level descriptors for the sample pack - when necessary.
$ Crawler -l low $PATH_TO_SAMPLE_PACK -o $PATH_TO_SAMPLE_PACK/features.db

# Load descriptors from $PATH_TO_SAMPLE_PACK and train the networks. Write 
# resulting binary shark model to the crawler's Resource/Models path, using
# the sample pack folder base-name as model filename.
$ ModelCreator $PATH_TO_SAMPLE_PACK/features.db 
```

## Testing models or adding new classes 

To test modified sample packs or when adding completely new categories, the `ModelTester` executable can be used to evaluate and debug the performance of the newly created model and classes. To do so, you can use the `Test%SAMPLE_PACK_NAME%Model.bat` scripts in the main AFEC main repository at [Scripts/ModelCreator](https://github.com/emuell/AFEC/tree/master/Scripts/ModelCreator).

Similar to the ModelCreator, they do the following:

```bash
# Create or update low level descriptors for the sample pack, when necessary.
$ Crawler -l low $PATH_TO_SAMPLE_PACK -o $PATH_TO_SAMPLE_PACK/features.db

# Load descriptors from PATH_TO_SAMPLE_PACK and train a simplified version of the model
# .
# When done, a summary for each class's prediction accuracy will be printed to the console. 
# Additionally it copies two files to the $PATH_TO_SAMPLE_PACK root folder:
# - Confusion.csv: the classification confusion matrix (see https://en.wikipedia.org/wiki/Confusion_matrix)
# - PredictionErrors.csv: a list of all audio files which have been mispredicted
$ ModelTester $PATH_TO_SAMPLE_PACK/features.db 
```

The prediction accuracy tells you how good each class/category performs in overall in the tests. Ideally all classes should have an accuracy of > 80% to be somewhat usable later on. 
`Confusion.csv` gives you more details about that, by also listing as which other categories the mispredicted samples got detected instead. 
`PredictionErrors.csv` shows more details about each single sound that got mispredicted, including all calculated class weights for all categories for this specific sample. 

`Confusion.csv` usually is a first good overview of how the model performs. `PredictionErrors.csv` then is useful to double-check a categories' (possibly modified) content in detail. If a sample often gets confused with some other category, this usually is a good hint that either the category can't be detected properly (ir isn't well defined) or that a specific sample simply is not a good example for the category: 
Ideally each sample pack's category folder should include a variety of prototype sounds that fit into this category, but all those prototypes should also not smear too much into other categories prototypes.

Note that when adding a completely new category, it's often necessary to double-check all old class prototypes again, to ensure that the old category prototypes are not represented by the new category as well. `Confusion.csv` again should give you a good hint here and `PredictionErrors.csv` what exactly seems to be the problem.
