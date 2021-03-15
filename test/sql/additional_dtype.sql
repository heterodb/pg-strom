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

CREATE TABLE various_dtypes(
    i1       int1,
    i2       int2,
    i4       int4,
    i8       int8,
    f2       float2,
    f4       float4,
    f8       float8,
    nm       numeric,
    ch       char(3)
);
INSERT INTO various_dtypes VALUES (11,12,14,18,21.1,22.2,24.4,33.33,'123');

-- memo: sql generating two liner
-- declare -A d=(["i2"]="int2" ["i4"]="int4" [i8]="int8" ["f2"]="float2" ["f4"]="float4" ["f8"]="float8" ["nm"]="numeric" ["ch"]="char(3)");
-- echo "SELECT " ; for cn in ${!d[@]}; do echo "cast($cn AS int1) \"${cn}_i1\", cast(i1 AS ${d[$cn]}) \"i1_$cn\"," ; done | awk '{if (NR==eof)print $0; else print $0","}' ; echo "FROM various_dtypes;"
SELECT 
cast(f2 AS int1) "f2_i1", cast(i1 AS float2) "i1_f2",
cast(f4 AS int1) "f4_i1", cast(i1 AS float4) "i1_f4",
cast(f8 AS int1) "f8_i1", cast(i1 AS float8) "i1_f8",
cast(ch AS int1) "ch_i1", cast(i1 AS char(3)) "i1_ch",
-- cast(nm AS int1) "nm_i1", cast(i1 AS numeric) "i1_nm",
cast(i8 AS int1) "i8_i1", cast(i1 AS int8) "i1_i8",
cast(i2 AS int1) "i2_i1", cast(i1 AS int2) "i1_i2",
cast(i4 AS int1) "i4_i1", cast(i1 AS int4) "i1_i4",
FROM various_dtypes;

---- comarison
-- eq,ne,lt,le,gt,ge
-- int1,int2,int4,int8

-- declare -A cs=(["eq"]="=" ["ne"]="<>" ["lt"]="<" ["le"]="<=" ["gt"]=">" ["ge"]=">=");
-- echo "SELECT " ; for c in ${!cs[@]}; do echo "i1 ${cs[$c]} i1 as i1_${c}_i1" ; done | awk -F, '{if (NR==eof)print $0; else print $0","}' ; echo "FROM various_dtypes;"
SELECT 
i1 = i1 as i1_eq_i1,
i1 >= i1 as i1_ge_i1,
i1 <> i1 as i1_ne_i1,
i1 > i1 as i1_gt_i1,
i1 <= i1 as i1_le_i1,
i1 < i1 as i1_lt_i1,
FROM various_dtypes;

-- unset d; declare -A d=(["i2"]="int2" ["i4"]="int4" [i8]="int8");
-- echo "SELECT "; for c in ${!cs[@]}; do for cn in ${!d[@]}; do echo "i1 ${cs[$c]} $cn \"i1_${c}_$cn\" , $cn ${cs[$c]} i1 \"${cn}_${c}_i1\"" ; done ; done | awk 'NR==1{print $0} NR>1{print ","$0}' ; echo "FROM various_dtypes;"
SELECT 
i1 = i8 "i1_eq_i8" , i8 = i1 "i8_eq_i1"
,i1 = i2 "i1_eq_i2" , i2 = i1 "i2_eq_i1"
,i1 = i4 "i1_eq_i4" , i4 = i1 "i4_eq_i1"
,i1 >= i8 "i1_ge_i8" , i8 >= i1 "i8_ge_i1"
,i1 >= i2 "i1_ge_i2" , i2 >= i1 "i2_ge_i1"
,i1 >= i4 "i1_ge_i4" , i4 >= i1 "i4_ge_i1"
,i1 <> i8 "i1_ne_i8" , i8 <> i1 "i8_ne_i1"
,i1 <> i2 "i1_ne_i2" , i2 <> i1 "i2_ne_i1"
,i1 <> i4 "i1_ne_i4" , i4 <> i1 "i4_ne_i1"
,i1 > i8 "i1_gt_i8" , i8 > i1 "i8_gt_i1"
,i1 > i2 "i1_gt_i2" , i2 > i1 "i2_gt_i1"
,i1 > i4 "i1_gt_i4" , i4 > i1 "i4_gt_i1"
,i1 <= i8 "i1_le_i8" , i8 <= i1 "i8_le_i1"
,i1 <= i2 "i1_le_i2" , i2 <= i1 "i2_le_i1"
,i1 <= i4 "i1_le_i4" , i4 <= i1 "i4_le_i1"
,i1 < i8 "i1_lt_i8" , i8 < i1 "i8_lt_i1"
,i1 < i2 "i1_lt_i2" , i2 < i1 "i2_lt_i1"
,i1 < i4 "i1_lt_i4" , i4 < i1 "i4_lt_i1"
FROM various_dtypes;


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