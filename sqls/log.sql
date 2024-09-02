WITH
  RAW_LOG AS (
  SELECT
    i.query_id,
    i.query AS keyword,
    i.index AS offset,
    i.event_timestamp AS impression_timestamp,
    i.document_id AS article_id,
    COALESCE(c.click, 0) AS click,
  FROM
    `karrot-data-production.karrotanalytics_v1_kr.client_impression_search_article` i
  LEFT JOIN (
    SELECT
      query_id,
      index,
      1 AS click
    FROM
      `karrot-data-production.karrotanalytics_v1_kr.client_click_search_article`
    WHERE
      DATE(event_timestamp, 'Asia/Seoul') = "{LOG_DATE}" ) c
  ON
    i.query_id = c.query_id
    AND i.index = c.index
  WHERE
    DATE(i.event_timestamp, 'Asia/Seoul') = "{LOG_DATE}" ),
  QUERY_STATS AS (
  SELECT
    keyword,
    COUNT(*) AS query_count
  FROM
    RAW_LOG
  GROUP BY
    keyword ),
  SESSION_STATS AS (
  SELECT
    query_id,
    SUM(click) AS sum_clicks,
    COUNT(*) AS session_length
  FROM
    RAW_LOG
  GROUP BY
    query_id )
SELECT
  RAW_LOG.*
FROM
  RAW_LOG
JOIN
  QUERY_STATS
ON
  RAW_LOG.keyword = QUERY_STATS.keyword
JOIN
  SESSION_STATS
ON
  RAW_LOG.query_id = SESSION_STATS.query_id
WHERE
  QUERY_STATS.query_count >= {MIN_QUERY_COUNT}
  AND SESSION_STATS.sum_clicks >= {MIN_CLICKS}
  AND SESSION_STATS.session_length >= {MIN_SESSION_LENGTH}
  AND SESSION_STATS.session_length <= {MAX_SESSION_LENGTH}