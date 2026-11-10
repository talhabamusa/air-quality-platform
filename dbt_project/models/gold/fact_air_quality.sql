select
    md5(p.sensor_id::text || '|' || p.hour_start_utc::text) as fact_key,
    p.hour_start_utc as hour_key,
    p.location_id as station_key,
    p.sensor_id,
    p.city as city_key,
    p.value as pm25_ugm3,
    p.value_raw as pm25_raw_ugm3,
    p.is_imputed,
    p.impute_method,
    case
        when p.impute_method = 'observed' then 'observed'
        when p.impute_method in ('linear', 'city_anchored') then 'imputed'
        when p.impute_method in ('city_scaled', 'linear_fallback') then 'imputed_low_confidence'
        else 'missing'
    end as quality_tier,
    p.gap_hours,
    p.pct_complete
from {{ source('staging', 'stg_pm25_hourly') }} as p