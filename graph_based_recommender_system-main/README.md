# graph_based_recommender_system

OMSCS CS 7643, project. a recommender system based on graph. recommends user product based on past product reviews.

## to setup environment

```bash
conda env create -f environment.yml
```

## to activate environment

```bash
conda activate GNNRS
```

or

```bash
source ./activate_environment.sh
```

## to export environment

```bash
source ./export_environment.sh
```


## to get benchmark datasets
```bash
sh get_benchmarks_dataset.sh
```

## to fetch data from mongodb for rdf
```bash
python generate_rdf.py --mongodb-url $MONGODB_URL
```



references:

data source:

- http://deepyeti.ucsd.edu/jianmo/amazon/index.html
- Justifying recommendations using distantly-labeled reviews and fined-grained aspects \
  Jianmo Ni, Jiacheng Li, Julian McAuley \
  Empirical Methods in Natural Language Processing (EMNLP), 2019
