# CompGCN
> 1. **FB15K-237 (Freebase 15K-237):** The FB15k dataset contains knowledge base relation triples and textual mentions of Freebase entity pairs. It has a total of 592,213 triplets with 14,951 entities and 1,345 relationships. FB15K-237 is a variant of the original dataset where inverse relations are removed, since it was found that a large number of test triplets could be obtained by inverting triplets in the training set.
> 2. **WN18RR (Wordnet 18rr):** is a link prediction dataset created from WN18, which is a subset of WordNet. WN18 consists of 18 relations and 40,943 entities. However, many text triples are obtained by inverting triples from the training set. Thus the WN18RR dataset is created to ensure that the evaluation dataset does not have inverse relation test leakage. In summary, WN18RR dataset contains 93,003 triples with 40,943 entities and 11 relation types.

# KGE
> 1. **CoDEx:** comprises a set of knowledge graph completion datasets extracted from Wikidata and Wikipedia that improve upon existing knowledge graph completion benchmarks in scope and level of difficulty. CoDEx comprises three knowledge graphs varying in size and structure, multilingual descriptions of entities and relations, and tens of thousands of hard negative triples that are plausible but verified to be false. 
> 2. **DBpedia:** a crowd-sourced community effort to extract structured content from the information created in various Wikimedia projects. DB100k is a subset of DBpedia. The main relation patterns are composition, inverse and subrelation. 
> 3. **FB15k (Freebase 15K):** The FB15k dataset contains knowledge base relation triples and textual mentions of Freebase entity pairs. It has a total of 592,213 triplets with 14,951 entities and 1,345 relationships. 
> 4. **kinship:** This relational database consists of 24 unique names in two families (they have equivalent structures).
> 5. **wikidata5m:** is a million-scale knowledge graph dataset with aligned corpus. This dataset integrates the Wikidata knowledge graph and Wikipedia pages. Each entity in Wikidata5m is described by a corresponding Wikipedia page, which enables the evaluation of link prediction over unseen entities.
> 6. **wn18:** The WN18 dataset has 18 relations scraped from WordNet for roughly 41,000 synsets, resulting in 141,442 triplets. It was found out that a large number of the test triplets can be found in the training set with another relation or the inverse relation. Therefore, a new version of the dataset WN18RR has been proposed to address this issue.
> 7. **wnrr:**  WNRR is constructed from WN18 by following similar procedure.
> 8. **yago3-10 (Yet Another Great Ontology 3-10):** YAGO3-10 is benchmark dataset for knowledge base completion. It is a subset of YAGO3 (which itself is an extension of YAGO) that contains entities associated with at least ten different relations. In total, YAGO3-10 has 123,182 entities and 37 relations, and most of the triples describe attributes of persons such as citizenship, gender, and profession.
> 9. **umls:** (lack of official information) Based on my understanding, it has been noted that the WN18 and FB15k datasets suffer from test set leakage, due to inverse relations from the training set being present in the test set, researchers introduced the dataset in "Convolutional 2D Knowledge Graph Embeddings", in which they investigated and validated several commonly used datasets -- deriving robust variants where necessary to ensure that models are evaluated on datasets where simply exploiting inverse relations cannot yield competitive results.
> 10. **wn11:** (lack of official information) It may be a variance of WordNet.
> 11. other datasets that are lack of information: **nations, toy**.

# KGAT
> 1. **last-fm:** consists of two kinds of data at the song level: tags and similar songs.
> 2. **yelp2018:** is a subset of Yelp's businesses, reviews, and user data. 
