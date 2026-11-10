select *
from {{ ref('fact_air_quality') }}
where quality_tier = 'observed'
  and pm25_ugm3 is distinct from pm25_raw_ugm3