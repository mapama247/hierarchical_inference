from hierarchical_inference import utils
import transformers
import torch
import os

try:
    from typing import List
except ImportError:
    from typing_extensions import List

class Classification:

    WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
    CACHE_DIR   = os.path.join(os.path.expanduser('~'), ".cache/huggingface/intelcomp")
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        self.models = utils.read_yaml(os.path.join(self.WORKING_DIR, "data/models.yaml"))
        self.classes = utils.read_yaml(os.path.join(self.WORKING_DIR, "data/classes.yaml"))
        self._avail_taxonomies = list(self.models.keys())

        print(f"Classifier ready to perform inference in the following taxonomies: {self._avail_taxonomies}")
        print(f"Device: {'GPU' if self.DEVICE=='cuda' else 'CPU'}")

        for taxonomy in self.models.keys():
            assert utils.check_consecutive(list(self.models[taxonomy].keys())), \
                (f"In models.yaml, the levels defined within {taxonomy} are not correct, "
                 f"they must be consecutive numbers. Please fix the YAML file.")
        for taxonomy in self.classes.keys():
            assert utils.check_consecutive(list(self.classes[taxonomy].keys())), \
                (f"In classes.yaml, the levels defined within {taxonomy} are not correct, "
                 f"they must be consecutive numbers. Please fix the YAML file.")

    @property
    def avail_taxonomies(self):
        return self._avail_taxonomies

    def update_taxonomies(self):
        self.__init__()

    def classify(self, taxonomy: str, text: str, verbose: bool =True) -> dict:
        """
        Classifies the given text using models trained in the given taxonomy.

        Parameters
        ----------
        taxonomy : str
            Name that refers to a set text classifiers which are organized in a hierarchical structure.
        text : str
            The input text to be classified by a cascade of text classifiers.
        verbose: bool
            Whether verbosity should be enabled or not. Default to True.

        Returns
        -------
        dict
            A dictionary where every element is the result obtained at a different level of the classification tree:
                - The key indicates the level of depth (0,1,2...) in the classification tree.
                - The value is a triplet that includes the class code, class name and confidence score.
        """
        if taxonomy in self.models:
            results = {}
            prev_level_class = None
            for level_str in sorted(self.models[taxonomy]):
                level = int(level_str)
                if level - 1 in results:
                    try:
                        prev_level_class = results[level-1][0]
                        model_path = self.models[taxonomy][level_str][prev_level_class]
                    except KeyError:
                        break  # reached a leaf of the classification tree
                else:
                    model_path = self.models[taxonomy][level_str]
                if verbose:
                    print(f"Running {taxonomy.upper()} classification at level {level}... (model: {model_path})")
                model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir=self.CACHE_DIR)
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, cache_dir=self.CACHE_DIR)
                pipe = transformers.pipeline(
                    task = "text-classification",
                    tokenizer = tokenizer,
                    model = model,
                    top_k = None,
                    device = self.DEVICE,
                )
                label_id   = pipe(text)[0][0]["label"]
                try:
                    label_name = self.classes[taxonomy][level_str][label_id] if level_str=="0" else self.classes[taxonomy][level_str][prev_level_class][label_id]
                except KeyError:
                    label_name = None
                score      = pipe(text)[0][0]["score"]
                results[level] = (label_id,label_name, score)
            utils.report_results(results)
        else:
            raise ValueError(f"The given taxonomy ({taxonomy}) is not supported. Available options: {list(self.models.keys())}")
        return results

    def classify_batch(self, taxonomy: str, texts: List[str], verbose=True) -> List[dict]:
        """
        Performs text classification on a batch of texts by recursively calling classify().

        Parameters
        ----------
        taxonomy : str
            Name that refers to a set text classifiers which are organized in a hierarchical structure.
        texts : List[st]
            List of input texts to be classified by a cascade of text classifiers.
        verbose: bool
            Whether verbosity should be enabled or not. Default to True.

        Returns
        -------
        List[dict]
            A list of dictionaries, with each containing the result obtained at a different level of the taxonomy:
                - The key indicates the level of depth (0,1,2...) in the classification tree.
                - The value is a triplet that includes the class code, class name and confidence score.
        """
        results = []
        for text in texts:
            result = self.classify(taxonomy, text, verbose)
            results.append(result)
        return results
