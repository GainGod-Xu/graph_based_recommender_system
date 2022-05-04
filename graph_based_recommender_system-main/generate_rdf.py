# %%
import argparse

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from util.mongodb import Mongo

# from util.neo4j import Neo4j


# %%


def get_triples(meta, reviews):
    # has reviewed relations
    # reviewerID has_reviewed productID
    triples = set()

    triples.add((meta.get('asin'), 'price', meta.get('price', '')))

    has_wrote_summary = set()
    summary_is_for = set()
    for review in reviews:
        if 'summary' in review:
            has_wrote_summary.add(
                (review["reviewerID"], "has_wrote_summary", review["summary"])
            )
            if meta['asin']:
                summary_is_for.add((review["summary"], "summary_is_for", meta["asin"]))
    triples.update(has_wrote_summary)

    ranked = set()
    for review in reviews:
        ranked.add((review["reviewerID"], f"ranked_{int(review['overall'])}", review["asin"]))
    triples.update(ranked)
        
    if "category" in meta:
        is_an_instance_of = set()
        for category in meta["category"]:
            is_an_instance_of.add((meta["asin"], "is_an_instance_of", category))
        triples.update(is_an_instance_of)
        
    if "also_buy" in meta:
        also_buy = set()
        for asin in meta["also_buy"]:
            also_buy.add((meta["asin"], "also_buy", asin))
        triples.update(also_buy)

    if "also_view" in meta:
        also_view = set()
        for asin in meta["also_view"]:
            also_view.add((meta["asin"], "also_view", asin))
        triples.update(also_view)

    # if 'image' in meta:
    #     has_such_number_of_images = set()
    #     has_such_number_of_images.add((meta['asin'], 'has_such_number_of_images', len(meta['image'])))
    #     triples.update(has_such_number_of_images)

    if "brand" in meta:
        is_of_brand = set()
        if meta['brand']:
            is_of_brand.add((meta["asin"], "is_of_brand", meta["brand"]))
        triples.update(is_of_brand)

    if "details" in meta:

        if "Discontinued by manufacturer:" in meta["details"]:
            is_discontinued = set()
            is_discontinued.add(
                (
                    meta["asin"],
                    "is_discontinued",
                    meta["details"]["Discontinued by manufacturer:"],
                )
            )
            triples.update(is_discontinued)
    return triples


# %%


def save(triples):
    
    # Make sure triple is of len 3, otherwise pd gets very slow
    t = {triple for triple in triples if type(triple) is tuple and len(triple)==3} 
    df = pd.DataFrame(t)
    df.dropna(inplace=True)
    df.to_csv(
        "data/amazon-product-review/full.txt", header=False, index=False, sep="\t"
    )

    # Define likes and dislikes
    full = pd.read_csv("data/amazon-product-review/full.txt", header=None, sep="\t")
    full.columns = ['h', 'l', 't']
    rat = full.loc[full['l'].isin(['ranked_5', 'ranked_4', 'ranked_2', 'ranked_1'])].copy()
    rat.loc[rat['l'].isin(['ranked_5', 'ranked_4']), 'l'] = 'likes'
    rat.loc[rat['l'].isin(['ranked_2', 'ranked_1']), 'l'] = 'dislikes'
    full = pd.concat([full, rat], axis=0).reset_index(drop=True)
    
    # Finds user and item appearances
    counter = full.loc[full['l'].isin(['has_wrote_summary', 'likes', 'dislikes'])].copy()
    user_count = counter.groupby('h').size()

    # Keep users and items with at least 5 user-item interactions
    user_kept = user_count[user_count >= 5].index.tolist()
    item_count = counter.groupby('t').size()
    item_kept = item_count[item_count >= 5].index.tolist()

    df = full.loc[((full['l'].isin(['has_wrote_summary', 'likes', 'dislikes'])) & 
                   (full['h'].isin(user_kept)) & (full['t'].isin(item_kept))) | 
                  (full['l'].isin(['also_buy', 'is_an_instance_of', 'is_of_brand', 'also_view']))].copy().reset_index(drop=True)
    df = df.loc[df[['h', 'l', 't']].notnull().all(axis=1)]

    lik = df.loc[df['l'].isin(['likes'])].copy()
    lik['usrgrp_idx'] = lik.groupby(['h']).cumcount()+1
    lik['itmgrp_idx'] = lik.groupby(['t']).cumcount()+1
    
    np.random.seed(1024)

    test_size = int(len(lik) * 0.1)
    df_test = lik.loc[(lik['usrgrp_idx'] > 2) & (lik['itmgrp_idx'] > 2)].sample(test_size)
    df_test.iloc[:,:3].to_csv("data/amazon-product-review/test.txt", header=False, index=False, sep="\t")

    # likes only data
    df_train = lik.loc[~lik.index.isin(df_test.index)].copy()
    df_valid = df_train.loc[(lik['usrgrp_idx'] > 1) & (lik['itmgrp_idx'] > 1)].sample(test_size)
    df_train = df_train.loc[~df_train.index.isin(df_valid.index)].copy()
    
    df_train.iloc[:,:3].to_csv("data/amazon-product-review/train_likes.txt", header=False, index=False, sep="\t")
    df_valid.iloc[:,:3].to_csv("data/amazon-product-review/valid_likes.txt", header=False, index=False, sep="\t")
        
    # knowledge graph data
    df_train = pd.concat([df.loc[~(df['l'].isin(['likes']))], lik.loc[~lik.index.isin(df_test.index)]], axis=0)
    df_train['h_count'] = df_train.groupby('h').cumcount()+1
    df_train['t_count'] = df_train.groupby('t').cumcount()+1

    valid_size = int(len(df_train) * 0.1)
    df_valid = df_train.loc[(df_train['h_count'] > 1) & (df_train['t_count'] > 1)].sample(valid_size)
    df_train = df_train.loc[~df_train.index.isin(df_valid.index)]
    
    df_train.to_csv("data/amazon-product-review/train_KG.txt", header=False, index=False, sep="\t")
    df_valid.to_csv("data/amazon-product-review/valid_KG.txt", header=False, index=False, sep="\t")
    df_train.iloc[:, :3].to_csv("data/amazon-product-review/train.txt", header=False, index=False, sep="\t")
    df_valid.iloc[:, :3].to_csv("data/amazon-product-review/valid.txt", header=False, index=False, sep="\t")
    
    # KGAT artifacts
    if not os.path.exists("data/amazon-product-review/kgat_data"):
        os.makedirs("data/amazon-product-review/kgat_data")
    
    user_list = pd.DataFrame(user_count.index.values).reset_index()[[0, 'index']]
    user_list.columns = ['org_id', 'remap_id']
    user_list.to_csv("data/amazon-product-review/kgat_data/user_list.txt", header=['org_id', 'remap_id'], index=False, sep=" ")
    item_list = pd.DataFrame(item_count.index.values).reset_index()[[0, 'index']]
    item_list.columns = ['org_id', 'remap_id']
    item_list.to_csv("data/amazon-product-review/kgat_data/item_list.txt", header=['org_id', 'remap_id'], index=False, sep=" ")
    entity_list = pd.DataFrame(set(df['h'].unique()).union(set(df['t'].unique()))).reset_index()[[0, 'index']]
    entity_list.columns = ['entity', 'entity_remap_id']
    entity_list.to_csv("data/amazon-product-review/kgat_data/entity_list.txt", header=['org_id', 'remap_id'], index=False, sep=" ")
    relation_list = pd.DataFrame(df.loc[df['l']!='likes', 'l'].unique()).reset_index()[[0, 'index']]
    relation_list.columns = ['l', 'l_id']
    relation_list.to_csv("data/amazon-product-review/kgat_data/relation_list.txt", header=['org_id', 'remap_id'], index=False, sep=" ")
    
    # df with alias
    kg = df.loc[df['l']!='likes'].copy()
    kg = kg.merge(entity_list, how='left', left_on='h', right_on='entity').drop(columns='entity')
    kg.rename(columns={'entity_remap_id': 'h_id'}, inplace=True)
    kg = kg.merge(entity_list, how='left', left_on='t', right_on='entity').drop(columns='entity')
    kg.rename(columns={'entity_remap_id': 't_id'}, inplace=True)
    kg = kg.merge(relation_list, how='left', on='l')
    kg[['h_id', 'l_id', 't_id']].to_csv("data/amazon-product-review/kgat_data/kg_final.txt", header=False, index=False, sep=" ")
    
    # Generate KG train, test, valid
    df_test = pd.read_csv("data/amazon-product-review/test.txt", header=None, sep="\t").iloc[:, :3]
    df_train = pd.read_csv("data/amazon-product-review/train_likes.txt", header=None, sep="\t").iloc[:, :3]
    df_valid = pd.read_csv("data/amazon-product-review/valid_likes.txt", header=None, sep="\t").iloc[:, :3]
    
    df_test.columns = ['h', 'l', 't']
    df_test = df_test.merge(user_list, how='left', left_on='h', right_on='org_id').rename(columns={'remap_id': 'h_id'})
    df_test = df_test.merge(item_list, how='left', left_on='t', right_on='org_id').rename(columns={'remap_id': 't_id'})
    kg_test_0 = df_test.groupby('h_id')['t_id'].unique()
    kg_test = pd.DataFrame(kg_test_0.tolist()).round().astype('Int64')
    kg_test.index = kg_test_0.index
    kg_test.to_csv("data/amazon-product-review/kgat_data/test.txt", header=False, sep=" ")
    
    df_train.columns = ['h', 'l', 't']
    df_train = df_train.merge(user_list, how='left', left_on='h', right_on='org_id').rename(columns={'remap_id': 'h_id'})
    df_train = df_train.merge(item_list, how='left', left_on='t', right_on='org_id').rename(columns={'remap_id': 't_id'})
    kg_train_0 = df_train.groupby('h_id')['t_id'].unique()
    kg_train = pd.DataFrame(kg_train_0.tolist()).round().astype('Int64')
    kg_train.index = kg_train_0.index
    kg_train.to_csv("data/amazon-product-review/kgat_data/train.txt", header=False, sep=" ")
    
    df_valid.columns = ['h', 'l', 't']
    df_valid = df_valid.merge(user_list, how='left', left_on='h', right_on='org_id').rename(columns={'remap_id': 'h_id'})
    df_valid = df_valid.merge(item_list, how='left', left_on='t', right_on='org_id').rename(columns={'remap_id': 't_id'})
    kg_valid_0 = df_valid.groupby('h_id')['t_id'].unique()
    kg_valid = pd.DataFrame(kg_valid_0.tolist()).round().astype('Int64')
    kg_valid.index = kg_valid_0.index
    kg_valid.to_csv("data/amazon-product-review/kgat_data/valid.txt", header=False, sep=" ")
    
# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongodb-host")
    parser.add_argument("--mongodb-port", default=27017)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default="amazon_product_review")
    args = parser.parse_args()
    print(args)
    mongo = Mongo(host=args.mongodb_host, port=args.mongodb_port, username=args.username, password=args.password, database=args.database)
    
    triples = set()
    
    query = {"main_cat": "All Beauty"}
    total = mongo.meta.count_documents(query)
    metas = mongo.get_meta(query, limit=total)
    
    def create_triples(meta):
        reviews = list(mongo.get_review_by_asin(meta["asin"]))
        triples.update(get_triples(meta, reviews))
        
    thread_map(create_triples, metas, max_workers=30, total=total)
   
    save(triples)


# %%
