---
--- test cases for custom data type int1
---
SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS custom_dtype_temp CASCADE;
CREATE SCHEMA custom_dtype_temp;
RESET client_min_messages;

SET search_path = custom_dtype_temp,public;

---
--- int1
---

--- data type define

CREATE TABLE int1_table(
    id  smallint,
    num1    int1
);

-- range check
INSERT INTO int1_table values (0,0);
INSERT INTO int1_table values (1,-128);  -- lower limit
INSERT INTO int1_table values (2,127);   -- upper limit
INSERT INTO int1_table values (3,-129);  -- lower error
INSERT INTO int1_table values (4,128);   -- upper error

-- cast
--- int1 -> int2
--- int1 -> int4
--- int1 -> int8
--- int1 -> float2
--- int1 -> float4
--- int1 -> float8
--- int1 -> numeric

---- comarison
-- eq,ne,lt,le,gt,ge
-- int1,int2,int4,int8

---- unary operators
--- +, - , @

---- arthmetic operators
--- +,-,*,/,%

---- bit operations
---- &,|,#,~,<<,>>

---- misc functions
-- money


---- aggregate function
-- sum,max,min,avg(larger,smaller)
-- variance,var_samp,var_pop,stddev, stddevv_samp,stddev_pop

---- index support
-- cmp,hash

-- cleanup temporary resource
SET client_min_messages = error;
DROP SCHEMA custom_dtype_temp CASCADE;