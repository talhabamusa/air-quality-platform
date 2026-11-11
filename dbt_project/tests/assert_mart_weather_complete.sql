select *
from {{ ref('mart_city_daily') }}
where weather_hours is distinct from 24