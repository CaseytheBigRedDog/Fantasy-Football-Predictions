-- Coverage check: of the players the model predicted, how many have a scored result?
-- A low scored_pct for a status group suggests many of those players did not play,
-- which means accuracy numbers for that group are affected by selection bias.
WITH preds AS (
    SELECT 3 AS week, player, team, position, status FROM predictions_2026_week3
    UNION ALL
    SELECT 4 AS week, player, team, position, status FROM predictions_2026_week4
),
scored AS (
    SELECT 3 AS week, player, team, position, actual FROM scored_2026_week3
    UNION ALL
    SELECT 4 AS week, player, team, position, actual FROM scored_2026_week4
)
SELECT
    COALESCE(p.status, 'none') AS practice_status,
    COUNT(*) AS predicted,
    SUM(CASE WHEN s.actual IS NOT NULL THEN 1 ELSE 0 END) AS scored,
    ROUND(100.0 * SUM(CASE WHEN s.actual IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS scored_pct,
    SUM(CASE WHEN s.actual = 0 THEN 1 ELSE 0 END) AS zero_point_games
FROM preds AS p
LEFT JOIN scored AS s
  ON s.week = p.week
 AND s.player = p.player
 AND s.team = p.team
 AND s.position = p.position
GROUP BY COALESCE(p.status, 'none')
ORDER BY predicted DESC;