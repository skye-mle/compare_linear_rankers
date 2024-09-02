SELECT
  CAST(id AS STRING) AS article_id,
  category_id,
  price,
  published_at,
FROM
  `karrotmarket.team_search_indexer_kr.articles_v2` doc
WHERE
  type = "FleaMarketArticle"
  AND destroyed_at IS NULL
  AND DATE(published_at, 'Asia/Seoul') BETWEEN "{DOC_START_DATE}" AND "{DOC_END_DATE}"