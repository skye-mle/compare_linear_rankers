import argparse
import yaml
import logging
import requests
from requests.auth import HTTPBasicAuth
import os
import pandas as pd
import pandas_gbq
from tqdm import tqdm
import json
import numpy as np
from metrics import evaluate_ranking_metrics


ES_INDEX_URL = "http://fleamarket-search-searching.kr.krmt.io/fleamarket-articles-v3"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_data_from_bigquery(saved_parquet_path):
    if os.path.exists(saved_parquet_path):
        logger.info(f"Saved data already exists. load data from {saved_parquet_path}")
        return pd.read_parquet(saved_parquet_path)

    with open("sqls/log.sql", "r", encoding="utf-8") as sql_file:
        log_sql = sql_file.read()
    log_sql = log_sql.format(
        LOG_DATE="2024-08-29",
        MIN_QUERY_COUNT=100,
        MIN_SESSION_LENGTH=10,
        MAX_SESSION_LENGTH=100,
        MIN_CLICKS=1,
    )
    with open("sqls/article.sql", "r", encoding="utf-8") as sql_file:
        article_sql = sql_file.read()
    article_sql = article_sql.format(
        DOC_START_DATE="2024-06-01", DOC_END_DATE="2024-08-29"
    )
    with open("sqls/keyword.sql", "r", encoding="utf-8") as sql_file:
        keyword_sql = sql_file.read()
    keyword_sql = keyword_sql.format()
    sql = f"""
    WITH 
        LOG AS ({log_sql}),
        ARTICLE AS ({article_sql}),
        KEYWORD AS ({keyword_sql})
    SELECT
        l.*,
        a.* EXCEPT (article_id),
        k.* EXCEPT (keyword) 
    FROM
        LOG l
    JOIN
        ARTICLE a
    ON
        l.article_id = a.article_id
    JOIN
        KEYWORD k
    ON
        l.keyword = k.keyword
    """
    df = pandas_gbq.read_gbq(
        sql, project_id="karrotmarket", dialect="standard", use_bqstorage_api=True
    )
    df.to_parquet(saved_parquet_path)
    return df


def get_text_match_dsl(keyword, doc_ids, text_field):
    return {
        "query": {
            "bool": {
                "must": {"match": {text_field: {"query": keyword}}},
                "filter": {"terms": {"id": doc_ids}},
            }
        },
        "_source": ["id"],
    }


def get_bm25_scores(keyword_to_docs):
    title_scores = []
    content_scores = []
    for keyword, doc_ids in tqdm(keyword_to_docs.items()):
        for sub_doc_ids in [
            doc_ids[i : i + 1000] for i in range(0, len(doc_ids), 1000)
        ]:
            try:
                dsl = get_text_match_dsl(keyword, sub_doc_ids, "title")
                response = requests.get(
                    f"{ES_INDEX_URL}/_search",
                    headers={"Content-Type": "application/json"},
                    json=dsl,
                    auth=HTTPBasicAuth("search", "search"),
                )
                hits = response.json()["hits"]["hits"]
                if hits:
                    title_scores.extend(
                        [
                            {
                                "keyword": keyword,
                                "article_id": d["_source"]["id"],
                                "bm25_title": d["_score"],
                            }
                            for d in hits
                        ]
                    )
            except:
                pass
            try:
                dsl = get_text_match_dsl(keyword, sub_doc_ids, "content")
                response = requests.get(
                    f"{ES_INDEX_URL}/_search",
                    headers={"Content-Type": "application/json"},
                    json=dsl,
                    auth=HTTPBasicAuth("search", "search"),
                )
                hits = response.json()["hits"]["hits"]
                if hits:
                    content_scores.extend(
                        [
                            {
                                "keyword": keyword,
                                "article_id": d["_source"]["id"],
                                "bm25_content": d["_score"],
                            }
                            for d in hits
                        ]
                    )
            except:
                pass
    title_scores = pd.DataFrame(title_scores)
    content_scores = pd.DataFrame(content_scores)
    bm25_scores = pd.merge(
        title_scores, content_scores, on=["keyword", "article_id"], how="inner"
    )
    bm25_scores.article_id = bm25_scores.article_id.astype(str)
    return bm25_scores


def get_category_match(df):
    category_match = []
    for doc_cat_id, cat_weights in zip(df["category_id"], df["category_weights"]):
        match = 0
        if not cat_weights:
            category_match.append(match)
            continue
        cat_weights = json.loads(cat_weights)
        for cat_weight in cat_weights:
            if doc_cat_id == cat_weight["category_id"] and cat_weight["is_boost"]:
                match = 1
        category_match.append(match)
    return category_match


def get_price_match(df):
    price_match = []
    for doc_price, price_ranges in zip(df["price"], df["price_ranges"]):
        match = 0
        if not price_ranges:
            price_match.append(match)
            continue
        price_ranges = json.loads(price_ranges)
        match = 0
        for price_range in price_ranges:
            price_range = price_range["price_range"]
            if price_range[0] <= doc_price and doc_price <= price_range[1]:
                match = 1
        price_match.append(match)
    return price_match


def gaussian_decay(interval_minute, offset=60, sigma=1200):
    return np.exp(
        -((np.maximum(0, np.abs(interval_minute) - offset)) ** 2) / (2 * sigma**2)
    )


def get_preprocessed_dataset(saved_data_path):
    if os.path.exists(saved_data_path):
        logger.info(
            f"Preprocessed dataset already exists. load data from {saved_data_path}"
        )
        df = pd.read_parquet(saved_data_path)
        return df

    logger.info("Load data from bigquery")
    df = load_data_from_bigquery("data/log.parquet")

    logger.info("Get BM25 scores")
    keyword_to_docs = (
        df[["keyword", "article_id"]]
        .drop_duplicates()
        .groupby("keyword")["article_id"]
        .apply(list)
        .to_dict()
    )
    bm25_scores = get_bm25_scores(keyword_to_docs)
    df = pd.merge(df, bm25_scores, how="inner", on=["keyword", "article_id"])

    logger.info("Filter invalid sessions")
    valid_query_ids = {
        k
        for k, v in df.groupby("query_id")["offset"].count().to_dict().items()
        if v > 1
    }
    df = df[df["query_id"].isin(valid_query_ids)]
    df["interval_minutes"] = (
        df["impression_timestamp"] - df["published_at"]
    ).dt.total_seconds() / 60
    df = df[df["interval_minutes"] >= 0]

    logger.info("Calculate features")
    df["category_match"] = get_category_match(df)
    df["price_match"] = get_price_match(df)
    df["title_match"] = (df["bm25_title"] > 0).astype(int)
    df["content_match"] = (df["bm25_content"] > 0).astype(int)
    df["recency"] = [
        gaussian_decay(interval_minute) for interval_minute in df["interval_minutes"]
    ]
    df = df[
        [
            "query_id",
            "keyword",
            "article_id",
            "click",
            "offset",
            "title_match",
            "content_match",
            "category_match",
            "recency",
            "price_match",
        ]
    ]
    df.to_parquet(saved_data_path)
    return df


def print_statistics(df):
    print("Number of search sessions:", len(df["query_id"].unique()))
    print(
        "Average length of sessions:", df.groupby("query_id")["offset"].count().mean()
    )
    print("Minimum length of sessions:", df.groupby("query_id")["offset"].count().min())
    print("Maximum length of sessions:", df.groupby("query_id")["offset"].count().max())
    print(
        "Average sum of clicks per session:",
        df.groupby("query_id")["label"].sum().mean(),
    )
    print(
        "Minimum sum of clicks per session:",
        df.groupby("query_id")["label"].sum().min(),
    )
    print(
        "Maximum sum of clicks per session:",
        df.groupby("query_id")["label"].sum().max(),
    )


def calculate_metrics(df, ks=[5, 10, 20, 50]):
    qid_keyword = dict(zip(df["query_id"], df["keyword"]))
    labels = df.groupby("query_id")["label"].apply(list).to_dict()
    scores = df.groupby("query_id")["score"].apply(list).to_dict()

    out_all = []
    out = (
        {f"P@{k}": [] for k in ks}
        | {f"R@{k}": [] for k in ks}
        | {f"N@{k}": [] for k in ks}
    )
    for query in tqdm(labels.keys()):
        label, score = labels[query], scores[query]
        metric = evaluate_ranking_metrics(score, label, ks, precision=4)
        for metric_name, metric_value in metric.items():
            out[metric_name].append(metric_value)
        metric["query_id"] = query
        metric["keyword"] = qid_keyword[query]
        out_all.append(metric)
    out = {k: np.mean(v) for k, v in out.items()}
    out["keyword"] = "global"
    out["query_id"] = "global"
    out_all.insert(0, out)
    return pd.DataFrame(out_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--price_boost", type=float, default=0.0)
    parser.add_argument("--exp_name", type=str, default="default")
    args = parser.parse_args()
    exp_dir = os.path.join("experiments", args.exp_name)
    if os.path.exists(exp_dir):
        logger.error(f"Experiment dir ({exp_dir}) already exists.")
        raise Exception
    else:
        os.makedirs(exp_dir)

    logger.info("Get feature and labels")
    df = get_preprocessed_dataset("data/preprocessed.parquet")
    df.rename(columns={"click": "label"}, inplace=True)
    df.sort_values(["query_id", "offset"], inplace=True)

    logger.info("Print statistics")
    print_statistics(df)

    logger.info("Calcuate scores")
    weights = {
        "category_match": 7,
        "title_match": 3,
        "content_match": 1,
    }

    df["relevance"] = (
        df["category_match"] * weights["category_match"]
        + df["title_match"] * weights["title_match"]
        + df["content_match"] * weights["content_match"]
    )
    df["score"] = df["relevance"] + df["recency"] + df["price_match"] * args.price_boost

    metric_out = calculate_metrics(df)
    metric_out.to_parquet(os.path.join(exp_dir, "metrics_all.parquet"))
    metric_by_kw = (
        metric_out.groupby("keyword")[
            ["P@5", "P@10", "P@20", "N@5", "N@10", "N@20", "R@5", "R@10", "R@20"]
        ]
        .mean()
        .reset_index()
    )
    metric_by_kw.to_csv(os.path.join(exp_dir, "metrics_by_keyword.csv"), index=False)
    logger.info(f"Result:\n {metric_by_kw.head(n=1)}")
