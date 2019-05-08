@ja{
<h1>関数と演算子</h1>
本章ではGPUデバイス上で実行可能な関数と演算子について説明します。
}
@en{
<h1>Functions and operators</h1>
This chapter introduces the functions and operators executable on GPU devices.
}

@ja:#型キャスト
@en:#Type cast

|destination type|source type|description|
|:---------------|:----------|:----------|
|`bool`|`int4`||
|`int2`|`int4,int8,float2,float4,float8,numeric`||
|`int4`|`int2,int8,float2,float4,float8,numeric`||
|`int8`|`int2,int4,float2,float4,float8,numeric`||
|`float2`|`int2,int4,int8,float4,float8,numeric`||
|`float4`|`int2,int4,int8,float2,float8,numeric`||
|`float8`|`int2,int4,int8,float2,float4,numeric`||
|`numeric`|`int2,int4,int8,float2,float4,float8`||
|`money`|`int4,int8,numeric`||
|`inet`|`cidr`|
|`date`|`timestamp,timestamptz`||
|`time`|`timetz,timestamp,timestamptz`||
|`timetz`|`time,timestamptz`||
|`timestamp`|`date,timestamptz`||
|`timestamptz`|`date,timestamp`||

@ja:#数値型演算子
@en:#Numeric functions/operators

|function/operator|description|
|:----------------|:----------|
|`TYPE COMP TYPE`    |Comparison of two values<br>`TYPE` is any of `int2,int4,int8`<br>`COMP` is any of `=,<>,<,<=,>=,>`|
|`TYPE COMP TYPE`    |Comparison of two values<br>`TYPE` is any of `float2,float4,float8`<br>`COMP` is any of `=,<>,<,<=,>=,>`|
|`numeric COMP numeric`|Comparison of two values<br>`COMP` is any of `=,<>,<,<=,>=,>`|
|`TYPE + TYPE`|Arithemetic addition<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|
|`TYPE - TYPE`|Arithemetic substract<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|
|`TYPE * TYPE`|Arithemetic multiplication<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|
|`TYPE / TYPE`|Arithemetic division<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|
|`TYPE % TYPE`|Reminer operator<br>`TYPE` is any of `int2,int4,int8`|
|`TYPE & TYPE`|Bitwise AND<br>`TYPE` is any of `int2,int4,int8`|
|<code>TYPE &#124; TYPE</code>|Bitwise OR<br>`TYPE` is any of `int2,int4,int8`|
|`TYPE # TYPE`|Bitwise XOR<br>`TYPE` is any of `int2,int4,int8`|
|`~ TYPE`     |Bitwise NOT<br>`TYPE` is any if `int2,int4,int8`|
|`TYPE >> int4`|Right shift<br>`TYPE` is any of `int2,int4,int8`|
|`TYPE << int4`|Left shift<br>`TYPE` is any of `int2,int4,int8`|
|`+ TYPE`     |Unary plus<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|
|`- TYPE`     |Unary minus<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|
|`@ TYPE`     |Absolute value<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|

@ja:#数学関数
@en:#Mathematical functions

|functions/operators|description|
|:------------------|:----------|
|`cbrt(float8)`     |cube root|
|`dcbrt(float8)`    |cube root|
|`ceil(float8)`     |nearest integer greater than or equal to argument|
|`ceiling(float8)`  |nearest integer greater than or equal to argument|
|`exp(float8)`      |exponential|
|`dexp(float8)`     |exponential|
|`floor(float8)`    |nearest integer less than or equal to argument|
|`ln(float8)`       |natural logarithm|
|`dlog1(float8)`    |natural logarithm|
|`log(float8)`      |base 10 logarithm|
|`dlog10(float8)`   |base 10 logarithm|
|`pi()`             |circumference ratio|
|`power(float8,float8)`|power|
|`pow(float8,float8)`  |power|
|`dpow(float8,float8)` |power|
|`round(float8)`    |round to the nearest integer|
|`dround(float8)`   |round to the nearest integer|
|`sign(float8)`     |sign of the argument|
|`sqrt(float8)`     |square root|
|`dsqrt(float8)`    |square root|
|`trunc(float8)`    |truncate toward zero|
|`dtrunc(float8)`   |truncate toward zero|

@ja:#三角関数
@en:#Trigonometric functions

|functions/operators|description|
|:------------------|:----------|
|`degrees(float8)`  |radians to degrees|
|`radians(float8)`  |degrees to radians|
|`acos(float8)`     |inverse cosine|
|`asin(float8)`     |inverse sine|
|`atan(float8)`     |inverse tangent|
|`atan2(float8,float8)`|inverse tangent of `arg1 / arg2`|
|`cos(float8)`      |cosine|
|`cot(float8)`      |cotangent|
|`sin(float8)`      |sine|
|`tan(float8)`      |tangent|

@ja:#日付/時刻型演算子
@en:#Date and time operators

|functions/operators|description|
|:------------------|:----------|
|`date COMP date`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`date COMP timestamp`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`date COMP timestamptz`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`time COMP time`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timetz COMP timetz`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timestamp COMP timestamp`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timestamp COMP date`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timestamptz COMP timestamptz`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timestamptz COMP date`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timestamp COMP timestamptz`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timestamptz COMP timestamp`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`interval COMP interval`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`date OP int4`|`OP` is either of `+,-`|
|`int4 + date`||
|`date - date`||
|`date + time`||
|`date + timetz`||
|`time + date`||
|`time - time`||
|`timestamp - timestamp`||
|`timetz OP interval`|`OP` is either of `+,-`|
|`timestamptz OP interval`|`OP` is either of `+,-`|
|`overlaps(TYPE,TYPE,TYPE,TYPE)`|`TYPE` is any of `time,timetz,timestamp,timestamptz`|
|`extract(text FROM TYPE)`|`TYPE` is any of `time,timetz,timestamp,timestamptz,interval`|
|`now()`||
|`- interval`|unary minus operator|
|`interval OP interval`|`OP` is either of `+,-`|


@ja:#文字列関数/演算子
@en:#Text functions/operators

|functions/operators|description|
|:------------------|:----------|
|`{text,bpchar} COMP {text,bpchar}`|`COMP` is either of `=,<>`|
|`{text,bpchar} COMP {text,bpchar}`|`COMP` is either of `<,<=,>=,>`<br>Only available on no-locale or UTF-8|
|`varchar || varchar`|Both side must be `varchar(n)` with maximum length.|
|`substring`, `substr`||
|`length(TYPE)`|length of the string<br>`TYPE` is either of `text,bpchar`|
|`TYPE LIKE text`|`TYPE` is either of `text,bpchar`|
|`TYPE NOT LIKE text`|`TYPE` is either of `text,bpchar`|
|`TYPE ILIKE text`|`TYPE` is either of `text,bpchar`<br>Only available on no-locale or UTF-8|
|`TYPE NOT ILIKE text`|`TYPE` is either of `text,bpchar`<br>Only available on no-locale or UTF-8|

@ja:#ネットワーク関数/演算子
@en:#Network functions/operators

|functions/operators|description|
|:------------------|:----------|
|`macaddr COMP macaddr`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`macaddr & macaddr`|Bitwise AND operator|
|<code>macaddr &#124; macaddr</code>|Bitwise OR operator|
|`~ macaddr`        |Bitwise NOT operator|
|`trunc(macaddr)`   |Set last 3 bytes to zero|
|`inet COMP inet`   |`COMP` is any of `=,<>,<,<=,>=,>`|
|`inet INCL inet`   |`INCL` is any of `<<,<<=,>>,>>=,&&`|
|`~ inet`           ||
|`inet & inet`      ||
|<code>inet &#124; inet</code>|
|`inet + int8`      ||
|`inet - int8`      ||
|`inet - inet`      ||
|`broadcast(inet)`  ||
|`family(inet)`     ||
|`hostmask(inet)`   ||
|`masklen(inet)`    ||
|`netmask(inet)`    ||
|`network(inet)`    ||
|`set_masklen(cidr,int)` ||
|`set_masklen(inet,int)` ||
|`inet_same_family(inet, inet)`||
|`inet_merge(inet,inet)`||



@ja:#通貨型演算子
@en:#Currency operators

|functions/operators|description|
|:------------------|:----------|
|`money COMP money` |`COMP` is any of `=,<>,<,<=,>=,>`|
|`money OP money`   |`OP` is any of `+,-,/`|
|`money * TYPE`     |`TYPE` is any of `int2,int4,float2,float4,float8`|
|`TYPE * money`     |`TYPE` is any of `int2,int4,float2,float4,float8`|
|`money / TYPE`     |`TYPE` is any of `int2,int4,float2,float4,float8`|

@ja:#uuid型演算子
@en:#UUID operators

|functions/operators|description|
|:------------------|:----------|
|`uuid COMP uuid`   |`COMP` is any of `=,<>,<,<=,>=,>`|

@ja:#JSONB型演算子
@en:#JSONB operators

|functions/operators    |description|
|:----------------------|:----------|
|`jsonb -> KEY`         |Get a JSON object field specified by the `KEY`|
|`jsonb -> NUM`         |Get a JSON array element indexed by `NUM`|
|`jsonb ->> KEY`        |Get a JSON object field specified by the `KEY`, as text|
|`jsonb ->> NUM`        |Get a JSON array element indexed by `NUM`|
|`(jsonb ->> KEY)::TYPE`|TYPE is any of `int2,int4,int8,float4,float8,numeric`<br>Get a JSON object field specified by `KEY`, as numeric data type. See the note below.|
|`(jsonb ->> NUM)::TYPE`|TYPE is any of `int2,int4,int8,float4,float8,numeric`<br>Get a JSON array element indexed by `NUM`, as numeric data type. See the note below.|
|`jsonb ? KEY`          |Check whether jsonb object contains the `KEY`|

@ja{
!!! Note
    `jsonb ->> KEY`演算子によって取り出した数値データを`float`や`numeric`など数値型に変換する時、通常、PostgreSQLはjsonb内部表現をテキストとして出力し、それを数値表現に変換するという2ステップの処理を行います。
    PG-Stromは`jsonb ->> KEY`演算子による参照とテキスト⇒数値表現へのキャストが連続している時、jsonbオブジェクトから数値表現を取り出すための特別なデバイス関数を使用する事で最適化を行います。
}
@en{
!!! Note
    When we convert a jsonb element fetched by `jsonb ->> KEY` operator into numerical data types like `float` or `numeric`, PostgreSQL takes 2 steps operations; an internal numerical form is printed as text first, then it is converted into numerical data type.
    PG-Strom optimizes the GPU code using a special device function to fetch a numerical datum from jsonb object/array, if `jsonb ->> KEY` operator and text-to-numeric case are continuously used.
}

@ja:#範囲型演算子
@en:#Range type functions/operators

|functions/operators|description|
|:------------------|:----------|
|`RANGE = RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE <> RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE < RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE <= RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE > RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE >= RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE @> RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE @> TYPE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`<br>`TYPE` is element type of `RANGE`.|
|`RANGE <@ RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`TYPE <@ RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`<br>`TYPE` is element type of `RANGE`.|
|`RANGE && RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE << RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE >> RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE &< RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE &> RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|<code>RANGE -&#124;- RANGE</code>  |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE + RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE * RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE - RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`lower(RANGE)`     |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`upper(RANGE)`     |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`isempty(RANGE)`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`lower_inc(RANGE)` |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`upper_inc(RANGE)` |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`lower_inf(RANGE)` |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`upper_inf(RANGE)` |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`range_merge(RANGE,RANGE)` |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|


@ja:#その他のデバイス関数
@en:#Miscellaneous device functions

|functions/operators|result|description|
|:------------------|:-----|:----------|
|`as_int8(float8)`  |`int8`  |Re-interpret double-precision floating point bit-pattern as 64bit integer value|
|`as_int4(float4)`  |`int4`  |Re-interpret single-precision floating point bit-pattern as 32bit integer value|
|`as_int2(float2)`  |`int2`  |Re-interpret half-precision floating point bit-pattern as 16bit integer value|
|`as_float8(int8)`  |`float8`|Re-interpret 64bit integer bit-pattern as double-precision floating point value|
|`as_float4(int4)`  |`float4`|Re-interpret 32bit integer bit-pattern as single-precision floating point value|
|`as_float2(int2)`  |`float2`|Re-interpret 16bit integer bit-pattern as half-precision floating point value|

