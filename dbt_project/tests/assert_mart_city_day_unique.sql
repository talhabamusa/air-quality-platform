select city_key, date_local, count(*) as n
from {{ ref('mart_city_daily') }}
group by 1, 2
having count(*) > 1