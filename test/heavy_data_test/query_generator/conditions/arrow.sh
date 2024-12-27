PARAMETERS_CONDITION="
SET pg_strom.gpudirect_enabled = OFF;
SET pg_strom.pinned_inner_buffer_threshold = 0;
"
ORDERS_TABLE_NAME="orders_arrow"