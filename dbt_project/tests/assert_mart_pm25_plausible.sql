select *
from {{ ref('mart_city_daily') }}
where pm25_mean_ugm3 < 0
   or pm25_mean_ugm3 > 1000