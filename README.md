# Hierarchical Text Classification System

Python package that allows using a cascade of text classifiers based on a given hierarchy of language models.

## Setup
- Intall: `pip install .`
- Update: `pip install --upgrade .`
- Uninstall: `pip uninstall hierarchical_inference`
- Create source distribution and wheel: `python setup.py sdist bdist_wheel`
- Check build after compilation: `twine check dist/*`

## Usage
Toy example commands:
```python
from hierarchical_inference import Classification

clf = Classification()
res = clf.classify(taxonomy="ipc", text="This is a test")
clf.classify("fos", "This is another test")
```
Output:
```commandline
Running IPC classification at level 0... (model: intelcomp/ipc_level0)
Running IPC classification at level 1... (model: intelcomp/ipc_level1_G)
####### LEVEL 0:
                - Class: G
                - Full class name:  PHYSICS
                - Confidence score: 0.9405
####### LEVEL 1:
                - Class: 09
                - Full class name:  EDUCATING; CRYPTOGRAPHY; DISPLAY; ADVERTISING; SEALS
                - Confidence score: 0.3386


Running FOS classification at level 0... (model: intelcomp/fos_level0)
####### LEVEL 0:
                - Class: Medicine
                - Full class name:  None
                - Confidence score: 0.5535
{0: ('Medicine', None, 0.5535141229629517)}
```
As it can be seen, the `classify()` method returns a dictionary object with the taxonomy levels as keys, 
where the values are triplets that contain the classification output.

Additionally, the method `classify_batch()` can be used in a similar manner to perform classification by batches,
giving it a list of texts rather than a single string to classify. Type `help(clf.classify_batch)` for more details.

### Customizable attributes

After instantiating an object from the Classification class, you can access the following attributes:

- `WORKING_DIR`: Location where the package files are located. Change only if you want to read YAML files from somewhere else.
- `CACHE_DIR`: Path where models will be cached for faster inference in subsequent calls. Defaults to `~/.cache/huggingface/intelcomp`.
- `DEVICE`: Either "cuda" or "cpu", based on the availability of GPUs.
- `models`: Dictionary with the paths to the different models, hierarchically organized. 
- `classes`: Dictionary to map class codes to class names, hierarchically organized as well.
- `avail_taxonomies`: List of taxonomies available in `data/models.yaml`. Notice that this is a private attribute that 
cannot be directly edited by the user, although it can be updated after modifying the YAML files.

## Adding/Removing new taxonomies

To add and/or remove new taxonomies, the YAML files within the `data` directory must be updated accordingly.
It is important to always follow the original structure, which is showcased in the next section.

The Classification object will be initialized based on the YAML files content at creation time, but if such files are
modified afterwards, it is possible to update the taxonomy details by simply calling the `update_taxonomies()` method.

```python
from hierarchical_inference import Classification

clf = Classification()
print(clf.avail_taxonomies) # ['fos', 'ipc', 'nace2']
# YAML FILES BEING MODIFIED...
clf.update_taxonomies()
print(clf.avail_taxonomies) # ['fos', 'ipc', 'nace2', 'NEW_TAXONOMY']
```

## YAML files
The `data` directory contains 2 YAML files that can be edited by the user to add new taxonomies:
### `models.yaml`: 
Lists the models to be used for each taxonomy, with a different indentation for each of its levels. \
Can either contain the names of models from the HuggingFace Hub or paths to local repositories. \
The first level (i.e. 0) of any taxonomy only contains a single model, which would be the root node of the 
classification tree, while deeper levels have a list of models for each possible outcome from the prior level. 
This is why, for the non-zero levels, it is important to make sure that the model key (value preceding ":") 
matches some label from the previous model in the chain.
### `classes.yaml`: 
Lists the possible outputs of all classifiers from `models.yaml`, following the same format. \
It basically maps class codes to their corresponding names, for a better interpretability of the results.

**Note**: \
It is important to ensure that the class names (keys of inner dictionaries) are read as strings. \
The reason is that some taxonomies (e.g. IPC) have numeric codes that are preceded by a number zero (e.g. "01"). \
In these cases, YAML reads the key as an integer and, thus, the preceding zero gets lost.
The library makes sure to convert all keys to strings for consistency, but these specific cases with leading zeros 
are hard to handle since there is no way to tell if there was a zero or not in the original file. \
In order to prevent that, please make sure to surround this kind of numeric codes with double quotes.

### Examples
Below are the default YAML files, for reference. Notice how no quotation marks are required at all in `models.yaml`, 
since none of the taxonomies contains "problematic" codes and the function used to process YAML files already reads all
keys as strings.
```yaml
# data/models.yaml

fos:
    0: intelcomp/fos_level0
ipc:
    0: intelcomp/ipc_level0
    1:
        A: intelcomp/ipc_level1_A
        B: intelcomp/ipc_level1_B
        C: intelcomp/ipc_level1_C
        D: intelcomp/ipc_level1_D
        E: intelcomp/ipc_level1_E
        F: intelcomp/ipc_level1_F
        G: intelcomp/ipc_level1_G
        H: intelcomp/ipc_level1_H
nace2:
    0: intelcomp/nace2_level0
    1:
        20: intelcomp/nace2_level1_20
        25: intelcomp/nace2_level1_25
        26: intelcomp/nace2_level1_26
        27: intelcomp/nace2_level1_27
        28: intelcomp/nace2_level1_28
        29: intelcomp/nace2_level1_29
        42: intelcomp/nace2_level1_42
```
```yaml
#data/classes.yaml

fos:
    0:
        0: ART
        1: BIOLOGY
        2: BUSINESS
        3: CHEMISTRY
        4: COMPUTERSCIENCE
        5: ECONOMICS
        6: ENGINEERING
        7: ENVIRONMENTALSCIENCE
        8: GEOGRAPHY
        9: GEOLOGY
        10: HISTORY
        11: MATERIALSSCIENCE
        12: MATHEMATICS
        13: MEDICINE
        14: PHILOSOPHY
        15: PHYSICS
        16: POLITICALSCIENCE
        17: PSYCHOLOGY
        18: SOCIOLOGY
nace2:
    0:
        10: MANUFACTURE OF FOOD PRODUCTS
        11: MANUFACTURE OF BEVERAGES
        12: MANUFACTURE OF TOBACCO PRODUCTS
        13: MANUFACTURE OF TEXTILES
        14: MANUFACTURE OF WEARING APPAREL
        [...]
    1:
        20:
            1: MANUFACTURE OF BASIC CHEMICALS, FERTILISERS AND NITROGEN COMPOUNDS
            2: MANUFACTURE OF PESTICIDES AND OTHER AGROCHEMICAL PRODUCTS
            3: MANUFACTURE OF PAINTS, VARNISHES AND SIMILAR COATINGS, PRINTING INK AND MASTICS
            4: MANUFACTURE OF SOAP AND DETERGENTS, CLEANING AND POLISHING PREPARATIONS
            5: MANUFACTURE OF OTHER CHEMICAL PRODUCTS
            6: MANUFACTURE OF MAN-MADE FIBRES
        25:
            1: MANUFACTURE OF STRUCTURAL METAL PRODUCTS
            2: MANUFACTURE OF TANKS, RESERVOIRS AND CONTAINERS OF METAL
            3: MANUFACTURE OF STEAM GENERATORS, EXCEPT CENTRAL HEATING HOT WATER BOILERS
            4: MANUFACTURE OF WEAPONS AND AMMUNITION
            5: FORGING, PRESSING, STAMPING AND ROLL-FORMING OF METAL; POWDER METALLURGY
            6: TREATMENT AND COATING OF METALS; MACHINING
            7: MANUFACTURE OF CUTLERY, TOOLS AND GENERAL HARDWARE
            9: MANUFACTURE OF OTHER FABRICATED METAL PRODUCTS
        26:
            1: MANUFACTURE OF ELECTRONIC COMPONENTS AND BOARDS
            2: MANUFACTURE OF COMPUTERS AND PERIPHERAL EQUIPMENT
            3: MANUFACTURE OF COMPUTERS AND PERIPHERAL EQUIPMENT
            4: MANUFACTURE OF CONSUMER ELECTRONICS
            5: MANUFACTURE OF INSTRUMENTS AND APPLIANCES FOR MEASURING, TESTING AND NAVIGATION
            6: MANUFACTURE OF IRRADIATION, ELECTROMEDICAL AND ELECTROTHERAPEUTIC EQUIPMENT
            7: MANUFACTURE OF OPTICAL INSTRUMENTS AND PHOTOGRAPHIC EQUIPMENT
            8: MANUFACTURE OF MAGNETIC AND OPTICAL MEDIA
        27:
            1: MANUFACTURE OF ELECTRIC MOTORS, GENERATORS, TRANSFORMERS
            2: MANUFACTURE OF BATTERIES AND ACCUMULATORS
            3: MANUFACTURE OF WIRING AND WIRING DEVICES
            4: MANUFACTURE OF ELECTRIC LIGHTING EQUIPMENT
            5: MANUFACTURE OF DOMESTIC APPLIANCES
            9: MANUFACTURE OF OTHER ELECTRICAL EQUIPMENT
        28:
            1: MANUFACTURE OF GENERAL — PURPOSE MACHINERY
            2: MANUFACTURE OF OTHER GENERAL-PURPOSE MACHINERY
            3: MANUFACTURE OF AGRICULTURAL AND FORESTRY MACHINERY
            4: MANUFACTURE OF METAL FORMING MACHINERY AND MACHINE TOOLS
            9: MANUFACTURE OF OTHER SPECIAL-PURPOSE MACHINERY
        29:
            1: MANUFACTURE OF MOTOR VEHICLES
            3: MANUFACTURE OF PARTS AND ACCESSORIES FOR MOTOR VEHICLES
        42:
            2: CONSTRUCTION OF UTILITY PROJECTS
            9: CONSTRUCTION OF OTHER CIVIL ENGINEERING PROJECTS
ipc:
    0:
        A: HUMAN NECESSITIES
        B: PERFORMING OPERATIONS, TRANSPORTING
        C: CHEMISTRY, METALLURGY
        D: TEXTILES, PAPER
        E: FIXED CONSTRUCTIONS
        F: MECHANICAL ENGINEERING, LIGHTING, HEATING, WEAPONS
        G: PHYSICS
        H: ELECTRICITY
    1:
        A:
            "01": AGRICULTURE; FORESTRY; ANIMAL HUSBANDRY; HUNTING; TRAPPING; FISHING
            "21": BAKING; EQUIPMENT FOR MAKING OR PROCESSING DOUGHS; DOUGHS FOR BAKING
            "22": BUTCHERING; MEAT TREATMENT; PROCESSING POULTRY OR FISH
            "23": FOODS OR FOODSTUFFS; TREATMENT THEREOF, NOT COVERED BY OTHER CLASSES
            "24": TOBACCO; CIGARS; CIGARETTES; SIMULATED SMOKING DEVICES; SMOKERS' REQUISITES
            [...]
        B:
            "01": PHYSICAL OR CHEMICAL PROCESSES OR APPARATUS IN GENERAL
            "02": CRUSHING, PULVERISING, OR DISINTEGRATING; PREPARATORY TREATMENT OF GRAIN FOR MILLING
            "03": SEPARATION OF SOLID MATERIALS USING LIQUIDS OR USING PNEUMATIC TABLES OR JIGS
            "04": CENTRIFUGAL APPARATUS OR MACHINES FOR CARRYING-OUT PHYSICAL OR CHEMICAL PROCESSES
            "05": SPRAYING OR ATOMISING IN GENERAL; APPLYING FLUENT MATERIALS TO SURFACES, IN GENERAL
            [...]
        C:
            "01": INORGANIC CHEMISTRY
            "02": TREATMENT OF WATER, WASTE WATER, SEWAGE, OR SLUDGE
            "03": GLASS; MINERAL OR SLAG WOOL
            "04": CEMENTS; CONCRETE; ARTIFICIAL STONE; CERAMICS; REFRACTORIES
            "05": FERTILISERS; MANUFACTURE THEREOF
            [...]
        D:
            "01": NATURAL OR MAN-MADE THREADS OR FIBRES; SPINNING
            "02": YARNS; MECHANICAL FINISHING OF YARNS OR ROPES; WARPING OR BEAMING
            "03": WEAVING
            "04": BRAIDING; LACE-MAKING; KNITTING; TRIMMINGS; NON-WOVEN FABRICS
            "05": SEWING; EMBROIDERING; TUFTING
            [...]
        E:
            "01": CONSTRUCTION OF ROADS, RAILWAYS, OR BRIDGES
            "02": HYDRAULIC ENGINEERING; FOUNDATIONS; SOIL-SHIFTING
            "03": WATER SUPPLY; SEWERAGE
            "04": BUILDING
            "05": LOCKS; KEYS; WINDOW OR DOOR FITTINGS; SAFES
            [...]
        F:
            "01": MACHINES OR ENGINES IN GENERAL; ENGINE PLANTS IN GENERAL; STEAM ENGINES
            "02": COMBUSTION ENGINES; HOT-GAS OR COMBUSTION-PRODUCT ENGINE PLANTS
            "03": MACHINES OR ENGINES FOR LIQUIDS; WIND, SPRING, OR WEIGHT MOTORS
            "04": POSITIVE-DISPLACEMENT MACHINES FOR LIQUIDS; PUMPS FOR LIQUIDS OR ELASTIC FLUIDS
            "15": FLUID-PRESSURE ACTUATORS; HYDRAULICS OR PNEUMATICS IN GENERAL
            [...]
        G:
            "01": MEASURING; TESTING
            "02": OPTICS
            "03": PHOTOGRAPHY; CINEMATOGRAPHY; ANALOGOUS TECHNIQUES USING WAVES OTHER THAN OPTICAL WAVES
            "04": HOROLOGY
            "05": CONTROLLING; REGULATING
            [...]
        H:
            "01": BASIC ELECTRIC ELEMENTS
            "02": GENERATION, CONVERSION, OR DISTRIBUTION OF ELECTRIC POWER
            "03": BASIC ELECTRONIC CIRCUITRY
            "04": ELECTRIC COMMUNICATION TECHNIQUE
            "05": ELECTRIC TECHNIQUES NOT OTHERWISE PROVIDED FOR
            [...]
```
