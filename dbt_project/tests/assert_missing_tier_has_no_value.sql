select *
from {{ ref('fact_air_quality') }}
where (quality_tier = 'missing' and pm25_ugm3 is not null)
   or (quality_tier <> 'missing' and pm25_ugm3 is null)