WITH CATEGORY AS (
  SELECT keyword,
        TO_JSON_STRING(ARRAY(SELECT STRUCT(
            cw.category_id,
            cw.is_boost
          )
          FROM UNNEST(category_weights) AS cw
        )) AS category_weights
  FROM `karrotmarket.team_search_data.fleamarket_category_weights`
  WHERE run_date = DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
),
PRICE AS (
  SELECT
    keyword as keyword,
    TO_JSON_STRING(ARRAY(
      SELECT
        STRUCT( value.value AS price_range)
      FROM
        UNNEST(
        VALUES
          ) AS value )) AS price_ranges
  FROM
    `karrotmarket.team_search_data.fleamarket_price_ranges`
  WHERE 
    date = DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
)
SELECT
  cat.keyword,
  cat.category_weights,
  price.price_ranges
FROM
  CATEGORY cat
FULL OUTER JOIN
  PRICE price
ON
  cat.keyword = price.keyword