@ja{
#関数と演算子
本章ではGPUデバイス上で実行可能な関数と演算子について説明します。
}
@en{
#Functions and operators
This chapter introduces the functions and operators executable on GPU devices.
}

@ja:##型キャスト
@en:##Type cast

- `bool`    <-- `int4`
- `int1`    <-- `int2`, `int4`, `int8`, `float2`, `float4`, `float8`, `numeric`
- `int2`    <-- `int1`, `int4`, `int8`, `float2`, `float4`, `float8`, `numeric`
- `int4`    <-- `bool`, `int1`, `int2`, `int8`, `float2`, `float4`, `float8`, `numeric`
- `int8`    <-- `int1`, `int2`, `int4`, `float2`, `float4`, `float8`, `numeric`
- `float2`  <-- `int1`, `int2`, `int4`, `int8`, `float4`, `float8`, `numeric`
- `float4`  <-- `int1`, `int2`, `int4`, `int8`, `float2`, `float8`, `numeric`
- `float8`  <-- `int1`, `int2`, `int4`, `int8`, `float2`, `float4`, `numeric`
- `numeric` <-- `int1`, `int2`, `int4`, `int8`, `float2`, `float4`, `float8`
- `money`   <-- `int4`, `int8`, `numeric`<br>
- `date`    <-- `timestamp`, `timestamptz`<br>
- `time`    <-- `timetz`, `timestamp`, `timestamptz`<br>
- `timetz`  <-- `time`, `timestamptz`<br>
- `timestamp` <-- `date`, `timestamptz`<br>
- `timestamptz` <-- `date`, `timestamp`


@ja:##数値型演算子
@en:##Numeric functions/operators

`bool COMP bool`
: @ja{論理値型の比較演算子。`COMP`は`=,<>`のいずれかです。}
: @en{comparison operators of boolean type. `COMP` is any of `=,<>`}


`INT COMP INT`
: @ja{整数型の比較演算子。<br>`INT`は`int1,int2,int4,int8`のいずれかで、左辺と右辺が異なる整数型であっても構いません。<br>`COMP`は`=,<>,<,<=,>=,>`のいずれかです。}
: @en{comparison operators of integer types.<br>`INT` is any of `int1,int2,int4,int8`. It is acceptable if left side and right side have different interger types.<br>`COMP` is any of `=,<>,<,<=,>=,>`}

`FP COMP FP`
: @ja{浮動小数点型の比較演算子。<br>`FP`は`float2,float4,float8`のいずれかで、左辺と右辺が異なる浮動小数点型であっても構いません。<br>`COMP`は`=,<>,<,<=,>=,>`のいずれかです。}
: @en{comparison operators of floating-point types.<br>`FP` is any of `float2,float4,float8`. It is acceptable if left side and right side have different floating-point types.<br>`COMP` is any of `=,<>,<,<=,>=,>`}

`numeric COMP numeric`
: @ja{実数型の比較演算子。<br>`COMP`は`=,<>,<,<=,>=,>`のいずれかです。}
: @en{comparison operators of `numeric` type.<br>`COMP` is any of `=,<>,<,<=,>=,>`}

`INT OP INT`
: @ja{整数型の算術演算子。<br>`INT`は`int1,int2,int4,int8`のいずれかで、左辺と右辺が異なる整数型であっても構いません。<br>`OP`は`+,-,*,/`のいずれかです。}
: @en{arithemetic operators of integer types.<br>`INT` is any of `int1,int2,int4,int8`. It is acceptable if left side and right side have different interger types.<br>`OP` is any of `+,-,*,/`}

`FP OP FP`
: @ja{浮動小数点型の算術演算子。<br>`FP`は`float2,float4,float8`のいずれかで、左辺と右辺が異なる浮動小数点型であっても構いません。<br>`COMP`は`+,-,*,/`のいずれかです。}
: @en{arithemetic operators of floating-point types.<br>`FP` is any of `float2,float4,float8`. It is acceptable if left side and right side have different floating-point types.<br>`COMP` is any of `+,-,*,/`}

`numeric OP numeric`
: @ja{実数型の比較演算子。<br>`OP`は`+,-,*,/`のいずれかです。}
: @en{comparison operators of `numeric` type.<br>`OP` is any of `+,-,*,/`}

`INT % INT`
: @ja{剰余演算子。`INT`は`int1,int2,int4,int8`のいずれかです。}
: @en{Reminer operator. `INT` is any of `int1,int2,int4,int8`}

`INT & INT`
: @ja{論理積演算子。`INT`は`int1,int2,int4,int8`のいずれかです。}
: @en{Bitwise AND operator. `INT` is any of `int1,int2,int4,int8`}

`INT | INT`
: @ja{論理和演算子。`INT`は`int1,int2,int4,int8`のいずれかです。}
: @en{Bitwise OR operator. `INT` is any of `int1,int2,int4,int8`}

`INT # INT`
: @ja{排他的論理和演算子。`INT`は`int1,int2,int4,int8`のいずれかです。}
: @en{Bitwise XOR operator. `INT` is any of `int1,int2,int4,int8`}

`~ INT`
: @ja{論理否定演算子。`INT`は`int1,int2,int4,int8`のいずれかです。}
: @en{Bitwise NOT operator. `INT` is any of `int1,int2,int4,int8`}

`INT >> int4`
: @ja{右シフト演算子。`INT`は`int1,int2,int4,int8`のいずれかです。}
: @en{Right shift operator. `INT` is any of `int1,int2,int4,int8`}

`INT << int4`
: @ja{左シフト演算子。`INT`は`int1,int2,int4,int8`のいずれかです。}
: @en{Left shift operator. `INT` is any of `int1,int2,int4,int8`}

`+ TYPE`
: @ja{単項プラス演算子。`TYPE`は`int1,int2,int4,int8,float2,float4,float8,numeric`のいずれかです。}
: @en{Unary plus operator. `TYPE` is any of `int1,int2,int4,int8,float2,float4,float8,numeric`.}

`- TYPE`
: @ja{単項マイナス演算子。`TYPE`は`int1,int2,int4,int8,float2,float4,float8,numeric`のいずれかです。}
: @en{Unary minus operator. `TYPE` is any of `int1,int2,int4,int8,float2,float4,float8,numeric`.}

`@  TYPE`
: @ja{絶対値。`TYPE`は`int1,int2,int4,int8,float2,float4,float8,numeric`のいずれかです。}
: @en{Absolute value. `TYPE` is any of `int1,int2,int4,int8,float2,float4,float8,numeric`.}

@ja:##数学関数
@en:##Mathematical functions

`float8 cbrt(float8)`<br>`float8 dcbrt(float8)`
:   cube root|

`float8 ceil(float8)`<br>`float8 ceiling(float8)`
:   nearest integer greater than or equal to argument

`float8 exp(float8)`<br>`float8 dexp(float8)`
:   exponential

`float8 floor(float8)`
:   nearest integer less than or equal to argument

`float8 ln(float8)`<br>`float8 dlog1(float8)`
:   natural logarithm

`float8 log(float8)`<br>`float8 dlog10(float8)`
:   base 10 logarithm

`float8 pi()`
:   circumference ratio

`float8 power(float8,float8)`<br>`float8 pow(float8,float8)`<br>`float8 dpow(float8,float8)`
:   power

`float8 round(float8)`<br>`float8 dround(float8)`
:   round to the nearest integer

`float8 sign(float8)`
:   sign of the argument

`float8 sqrt(float8)`<br>`float8 dsqrt(float8)`
:   square root|

`float8 trunc(float8)`<br>`float8 dtrunc(float8)`
:   truncate toward zero|

@ja:##三角関数
@en:##Trigonometric functions

`float8 degrees(float8)`
: @ja{ラジアンに対応する度}
: @en{radians to degrees}

`float8 radians(float8)`
: @ja{度に対応するラジアン}
: @en{degrees to radians}

`float8 acos(float8)`
: @ja{逆余弦関数}
: @en{inverse cosine}

`float8 asin(float8)`
: @ja{逆正弦関数}
: @en{inverse sine}

`float8 atan(float8)`
: @ja{逆正接関数}
: @en{inverse tangent}

`float8 atan2(float8,float8)`
: @ja{`arg1 / arg2`の逆正接関数}
: @en{inverse tangent of `arg1 / arg2`}

`float8 cos(float8)`
: @ja{余弦関数}
: @en{cosine}

`float8 cot(float8)`
: @ja{余接関数}
: @en{cotangent}

`float8 sin(float8)`
: @ja{正弦関数}
: @en{sine}

`float8 tan(float8)`
: @ja{正接関数}
: @en{tangent}

@ja:##日付/時刻型演算子
@en:##Date and time operators

`date COMP date`
: @ja{`date`型同士の比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれか。}
: @en{comparison operators for `date` type. `COMP` is any of `=,<>,<,<=,>=,>`.}

`date COMP timestamp`<br>`timestamp COMP date`
: @ja{`date`型と`timestamp`型の比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれか。}
: @en{comparison operators for `date` and `timestamp` type. `COMP` is any of `=,<>,<,<=,>=,>`.}

`date COMP timestamptz`<br>`timestamptz COMP date`
: @ja{`date`型と`timestamptz`型の比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれか。}
: @en{comparison operators for `date` and `timestamptz` type. `COMP` is any of `=,<>,<,<=,>=,>`.}

`time COMP time`
: @ja{`time`型同士の比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれか。}
: @en{comparison operators for `time` type. `COMP` is any of `=,<>,<,<=,>=,>`.}

`timetz COMP timetz`
: @ja{`timetz`型同士の比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれか。}
: @en{comparison operators for `timetz` type. `COMP` is any of `=,<>,<,<=,>=,>`.}

`timestamp COMP timestamp`
: @ja{`timestamp`型同士の比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれか。}
: @en{comparison operators for `timestamp` type. `COMP` is any of `=,<>,<,<=,>=,>`.}

`timestamptz COMP timestamptz`
: @ja{`timestamptz`型同士の比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれか。}
: @en{comparison operators for `timestamptz` type. `COMP` is any of `=,<>,<,<=,>=,>`.}

`timestamp COMP timestamptz`<br>`timestamptz COMP timestamp`
: @ja{`timestamp`型と`timestamptz`型の比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれか。}
: @en{comparison operators for `timestamp` and `timestamptz` type. `COMP` is any of `=,<>,<,<=,>=,>`.}

`interval COMP interval`
: @ja{`interval`型同士の比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれか。}
: @en{comparison operators for `interval` type. `COMP` is any of `=,<>,<,<=,>=,>`.}

`date + int4`<br>`int4 + date`
: @ja{`date`型の加算演算子}
: @en{addition operator of `date` type}

`date - int4`
: @ja{`date`型の減算演算子}
: @en{subtraction operator of `date` type}

`date - date`
: @ja{`date`型同士の差分}
: @en{difference between `date` types}

`date + time`<br>`time + date`
: @ja{`date`と`time`から`timestamp`を生成します}
: @en{constructs a `timestamp` from `date` and `time`}

`date + timetz`
: @ja{`date`と`timetz`から`timestamptz`を生成します}
: @en{constructs a `timestamptz` from `date` and `timetz`}

`time - time`
: @ja{`time`型同士の差分}
: @en{difference between `time` types}

`timestamp - timestamp`
: @ja{`timestamp`型同士の差分}
: @en{difference between `timestamp` types}

`timetz + interval`<br>`timetz - interval`
: @ja{`timetz`と`interval`を加算、または減算します。}
: @en{addition or subtraction operator of `timetz` by `interval`.}

`timestamptz + interval`<br>`timestamptz - interval`
: @ja{`timestamptz`と`interval`を加算、または減算します。}
: @en{addition or subtraction operator of `timestamptz` by `interval`.}

`overlaps(TYPE,TYPE,TYPE,TYPE)`
: @ja{2つの時間間隔が重なるかどうかを判定します。<br>`TYPE`は`time,timetz,timestamp,timestamptz`のいずれか一つです。}
: @en{checks whether the 2 given time periods overlaps.<br>`TYPE` is any of `time,timetz,timestamp,timestamptz`.}

`extract(text FROM TYPE)`
: @ja{`day`や`hour`など日付時刻型の部分フィールドの抽出。<br>`TYPE`は`time,timetz,timestamp,timestamptz,interval`のいずれか一つです。}
: @en{retrieves subfields such as `day` or `hour` from date/time values.<br>`TYPE` is any of `time,timetz,timestamp,timestamptz,interval`.}

`now()`
: @ja{トランザクションの現在時刻}
: @en{current time of the transaction}

`- interval`
: @ja{`interval`型の単項マイナス演算子}
: @en{unary minus operator of `interval` type}

`interval + interval`
: @ja{`interval`型の加算演算子}
: @en{addition operator of `interval` type}

`interval - interval`
: @ja{`interval`型の減算演算子}
: @en{subtraction operator of `interval` type}

@ja:##文字列関数/演算子
@en:##Text functions/operators

`{text,bpchar} COMP {text,bpchar}`
: @ja{比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれかです。<br>なお、`<,<=,>=,>`演算子はロケール設定がUTF-8またはC(ロケール設定なし)の場合にのみ有効です。}
: @en{comparison operators; `COMP` is any of `=,<>,<,<=,>=,>`<br>Note that `<,<=,>=,>` operators are valid only when locale is UTF-8 or C (no locale).}

`varchar || varchar`
: @ja{文字列結合<br>結果文字列の最大長を予測可能とするため、両辺は`varchar(n)`でなければいけません。}
: @en{concatenates the two strings.<br>Both side must be `varchar(n)` to ensure maximum length of the result being predictible.}

`substring(text,int4)`<br>`substring(text,int4,int4)`<br>`substr(text,int4)`<br>`substr(text,int4,int4)`
: @ja{部分文字列の切り出し}
: @en{extracts the substring}

`length({text,bpchar})`
: @ja{文字列長}
: @en{length of the string}

`{text,bpchar} [NOT] LIKE text`
: @ja{LIKE表現を用いたパターンマッチング}
: @en{pattern-matching according to the LIKE expression}

`{text,bpchar} [NOT] ILIKE text`
: @ja{LIKE表現を用いた大文字小文字を区別しないパターンマッチング。<br>なお、`ILIKE`演算子はロケール設定がUTF-8またはC(ロケール設定なし)の場合にのみ有効です。}
: @en{case-insensitive pattern-matching according to the LIKE expression.<br>Note that `ILIKE` operator is valid only when locale is UTF-8 or C (no locale).}

@ja:##ネットワーク関数/演算子
@en:##Network functions/operators

`macaddr COMP macaddr`
: @ja{比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれかです。}
: @en{comparison operators; `COMP` is any of `=,<>,<,<=,>=,>`}

`macaddr & macaddr`
: @ja{ビット積演算子}
: @en{Bitwise AND operator}

`macaddr | macaddr`
: @ja{ビット和演算子}
: @en{Bitwise OR operator}

`~ macaddr`
: @ja{ビット否定演算子}
: @en{Bitwise NOT operator}

`trunc(macaddr)`
: @ja{末尾の3バイトをゼロに設定する}
: @en{Set last 3 bytes to zero}

`inet COMP inet`
: @ja{比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれかです。}
: @en{comparison operators; `COMP` is any of `=,<>,<,<=,>=,>`}

`inet << inet`
: @ja{左辺は右辺に内包される。}
: @en{Left side is contained by right side}

`inet <<= inet`
: @ja{左辺は右辺に内包されるか等しい。}
: @en{Left side is contained by or equals to right side}

`inet >> inet`
: @ja{左辺は右辺を内包する。}
: @en{Left side contains right side}

`inet >>= inet`
: @ja{左辺は右辺を内包するか等しい。}
: @en{Left side contains or is equals to right side}

`inet && inet`
: @ja{左辺は右辺を内包するか内包される}
: @en{Left side contains or is contained by right side}

`~ inet`
: @ja{ビット否定演算子}
: @en{Bitwise NOT operator}

`inet & inet`
: @ja{ビット積演算子}
: @en{Bitwise AND operator}

`inet | inet`
: @ja{ビット和演算子}
: @en{Bitwise OR operator}

`inet + int8`
: @ja{加算演算子}
: @en{addition operator}

`inet - int8`
: @ja{減算演算子}
: @en{subtraction operator}

`inet - inet`
: @ja{減算演算子}
: @en{subtraction operator}

`broadcast(inet)`
: @ja{ネットワークアドレスのブロードキャストアドレスを返す}
: @en{returns the broadcast address of the given network address}

`family(inet)`
: @ja{ネットワークアドレスのアドレスファミリを返す。IPv4の場合は`4`、IPv6の場合は`6`}
: @en{returns the family of the given network address; `4` for IPv4, and `6` for IPv6}

`hostmask(inet)`
: @ja{ネットワークアドレスのホストマスクを返す}
: @en{extract host mask of the given network address}

`masklen(inet)`
: @ja{ネットワークアドレスのマスク長を返す}
: @en{extract netmask length of the given network address}

`netmask(inet)`
: @ja{ネットワークアドレスのネットマスクを返す}
: @en{extract netmask of the given network address}

`network(inet)`
: @ja{ネットワークアドレスのネットワーク部を返す}
: @en{extract network part of the given network address}

`set_masklen(NETADDR,int)`
: @ja{ネットワークアドレスのネットマスク長を設定する。`NETADDR`は`inet`か`cidr`のどちらか。}
: @en{set netmask length of the given network address; `NETADDR` is either `inet` or `cidr`.}

`inet_merge(inet,inet)`
: @ja{両方のネットワークを含む最小のネットワークを返す}
: @en{the smallest network which includes both of the given networks}

@ja:##通貨型演算子
@en:##Currency operators

`money COMP money`
: @ja{比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれかです。}
: @en{comparison operators; `COMP` is any of `=,<>,<,<=,>=,>`}

`money OP money`
: @ja{算術演算子。`OP`は`+,-,/`のいずれかです。}
: @en{arthmetric operators; `OP` is any of `+,-,/`}

`money * TYPE`<br>`TYPE * money`
: @ja{通貨型と数値型の乗算。`TYPE`は`int1,int2,int4,float2,float4,float8`のいずれかです。}
: @en{Multiply a currency with a numeric value; `TYPE` is any of `int2,int4,float2,float4,float8`}

`money / TYPE`
: @ja{通貨型の数値型による除算。`TYPE`は`int1,int2,int4,float2,float4,float8`のいずれかです。}
: @en{Division of a currency by a numeric value; `TYPE` is any of `int2,int4,float2,float4,float8`}

`money / money`
: @ja{通貨型同士の除算。}
: @en{Division of currency values}

@ja:##uuid型演算子
@en:##UUID operators

`uuid COMP uuid`
: @ja{比較演算子。`COMP`は`=,<>,<,<=,>=,>`のいずれかです。}
: @en{comparison operator. `COMP` is any of `=,<>,<,<=,>=,>`}

@ja:##JSONB型演算子
@en:##JSONB operators

`jsonb -> KEY`
: @ja{`KEY`で指定されたJSONオブジェクトフィールドを取得する}
: @en{Get a JSON object field specified by the `KEY`}

`jsonb -> NUM`
: @ja{インデックス番号`NUM`で指定されたJSONオブジェクトフィールドを取得する}
: @en{Get a JSON array element indexed by `NUM`}

`jsonb ->> KEY`
: @ja{`KEY`で指定されたJSONオブジェクトフィールドをテキスト値として取得する}
: @en{Get a JSON object field specified by the `KEY`, as text}

`jsonb ->> NUM`
: @ja{インデックス番号`NUM`で指定されたJSONオブジェクトフィールドをテキスト値として取得する}
: @en{Get a JSON array element indexed by `NUM`, as text}

`(jsonb ->> KEY)::TYPE`
: @ja{`TYPE`が`int2,int4,int8,float4,float8,numeric`のいずれかである場合、`KEY`で指定されたJSONオブジェクトフィールドを数値型として取得する。下記の補足も参照。}
: @en{If `TYPE` is any of `int2,int4,int8,float4,float8,numeric`, get a JSON object field specified by `KEY`, as numeric data type. See the note below.}

`(jsonb ->> NUM)::TYPE`
: @ja{`TYPE`が`int2,int4,int8,float4,float8,numeric`のいずれかである場合、インデックス番号`NUM`で指定されたJSONオブジェクトフィールドを数値型として取得する。下記の補足も参照。}
: @en{If `TYPE` is any of `int2,int4,int8,float4,float8,numeric`<br>Get a JSON array element indexed by `NUM`, as numeric data type. See the note below.}

`jsonb ? KEY`
: @ja{jsonbオブジェクトが指定された`KEY`を含むかどうかをチェックする}
: @en{Check whether jsonb object contains the `KEY`}

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

@ja:##範囲型演算子
@en:##Range type functions/operators

@ja{
!!! Note
    以下の説明において、`RANGE`は任意の{`int4range`、`int8range`、`tsrange`、`tstzrange`、`daterange`}です。<br>
    以下の説明において、`TYPE`は同時に使われている`RANGE`の要素型です。
}
@en{
!!! Note
    `RANGE` is any of {`int4range`, `int8range`, `tsrange`, `tstzrange`, `daterange`} in this section.<br>
    `TYPE` is the element type of the `RANGE` which is introduced together in this section.
}

`RANGE = RANGE`
: @ja{両辺が等しい} @en{Both sides are equal.}

`RANGE <> RANGE`
: @ja{両辺が等しくない} @en{Both sides are not equal.}

`RANGE < RANGE`
: @ja{左辺は右辺より小さい} @en{Left side is less than right side.}

`RANGE <= RANGE`
: @ja{左辺は右辺より小さいか等しい}
: @en{Left side is less than or equal to right side.}

`RANGE > RANGE`
: @ja{左辺は右辺より大きい}
: @en{Left side is greater than right side.}

`RANGE >= RANGE`
: @ja{左辺は右辺より大きいか等しい}
: @en{Left side is greater than or equal to right side.}

`RANGE @> RANGE`
: @ja{左辺の範囲は右辺の範囲を包含する}
: @en{The range in left side contains the range in right side.}

`RANGE @> TYPE`
: @ja{左辺の範囲は右辺の要素を包含する}
: @en{The range in left side contains the element in right side.}

`RANGE <@  RANGE`
: @ja{左辺の範囲は右辺の範囲に包含される}
: @en{The range in left side is contained by the range in right side.}

`TYPE <@  RANGE`
: @ja{左辺の要素は右辺の範囲に包含される}
: @en{The element in left side is contained by the range in right side.}

`RANGE && RANGE`
: @ja{左辺と右辺は重複する（共通点を持つ）}
: @en{Left and right side are overlap (they have points in common).}

`RANGE << RANGE`
: @ja{左辺は厳密に右辺よりも小さい}
: @en{Left side is strictly less than the right side}

`RANGE >> RANGE`
: @ja{左辺は厳密に右辺よりも大きい}
: @en{Left side is strictly greater than the right side}

`RANGE &< RANGE`
: @ja{左辺のいかなる点も右辺の範囲を越えない}
: @en{Any points in the left side is never greater than the right side}

`RANGE &> RANGE`
: @ja{右辺のいかなる点も左辺の範囲を越えない}
: @en{Any points in the right side is never greater than the left side}

`RANGE -|- RANGE`
: @ja{左辺と右辺は隣接している}
: @en{Left side is adjacent to the right side}

`RANGE + RANGE`
: @ja{左辺と右辺による結合範囲を返す}
: @en{A union range by the left side and right side}

`RANGE * RANGE`
: @ja{左辺と右辺による交差範囲を返す}
: @en{An intersection range by the left and right side}

`RANGE - RANGE`
: @ja{左辺と右辺による差分範囲を返す}
: @en{An difference range by the left and right side}

`lower(RANGE)`
: @ja{範囲の下限を返す}
: @en{lower bound of the range}

`upper(RANGE)`
: @ja{範囲の上限を返す}
: @en{upper bound of the range}

`isempty(RANGE)`
: @ja{空の範囲かどうかをチェックする}
: @en{checks whether the range is empty}

`lower_inc(RANGE)`
: @ja{下限は内包されているかどうかをチェックする}
: @en{checks whether the lower bound is inclusive}

`upper_inc(RANGE)`
: @ja{上限は内包されているかをチェックする}
: @en{checks whether the upper bound is inclusive}

`lower_inf(RANGE)`
: @ja{下限は無限大かどうかをチェックする}
: @en{checks whether the lower bound is infinite}

`upper_inf(RANGE)`
: @ja{上限は無限大かどうかをチェックする}
: @en{checks whether the upper bound is infinite}

`range_merge(RANGE,RANGE)`
: @ja{両方の範囲を含む最小の範囲を返す}
: @en{returns the smallest range which includes both of the given ranges}


@ja:##PostGIS関数
@en:##PostGIS Functions

`geometry st_makepoint(float8,float8)`<br>`geometry st_point(float8,float8)`
: @ja{2次元座標を含むPOINT型ジオメトリを返す}
: @en{It makes 2-dimensional POINT geometry.}

`geometry st_makepoint(float8,float8,float8)`
: @ja{3次元座標を含むPOINT型ジオメトリを返す}
: @en{It makes 3-dimensional POINT geometry.}

`geometry st_makepoint(float8,float8,float8,float8)`
: @ja{ZM座標を含むPOINT型ジオメトリを返す}
: @en{It makes 4-dimensional POINT geometry.}

`geometry st_setsrid(geometry,int4)`
: @ja{ジオメトリにSRIDを設定する}
: @en{It assigns SRID on the given geometry}

`float8 st_distance(geometry,geometry)`
: @ja{ジオメトリ間の距離を`float8`で返す}
: @en{It returns the distance between geometries in `float8`.}

`bool st_dwithin(geometry,geometry,float8)`
: @ja{ジオメトリ間の距離が指定値以内なら真を返す。`st_distance`と比較演算子の組み合わせよりも高速な場合がある。}
: @en{It returns `true` if the distance between geometries is shorter than the specified threshold. It is often faster than the combination of `st_distance` and comparison operator.}

`text st_relate(geometry,geometry)`
: @ja{ジオメトリ間の交差状態を判定し、DE9-IM(Dimensionally Extended Nine-Intersection Matrix)書式を返す。}
: @en{It checks intersection of geometries, then returns DE9-IM(Dimensionally Extended Nine-Intersection Matrix) format string.}

`bool st_contains(geometry,geometry)`
: @ja{ジオメトリ1がジオメトリ2を包含する時、真を返す。}
: @en{It returns whether the geometry1 fully contains the geometry1.}

`bool st_crosses(geometry,geometry)`
: @ja{ジオメトリ同士が空間的に交差する時、真を返す。}
: @en{It returns whether the geometries are crossed.}

`int4 st_linecrossingdirection(geometry,geometry)`
: @ja{2つのLINESTRING型ジオメトリがどのように交差するか（しないか）を返す。}
: @en{It checks how two LINESTRING geometries are crossing, or not crossing. }

