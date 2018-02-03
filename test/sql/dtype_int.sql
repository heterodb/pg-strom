---
--- Test cases for integer data types
---
--
-- int2
--
RESET pg_strom.enabled;
SELECT a, b
  INTO pg_temp.test_s01a
  FROM t_int2
 WHERE a > b AND @b > a;
SELECT a,b,-a v1, @b v2
  INTO pg_temp.test_s02a
  FROM t_int2
 WHERE (a < b AND @a > b) OR a = b;
SELECT a, b, a::int + b::int v1, a::int - b::int v2, a::int * b::int v3
  INTO pg_temp.test_s03a
  FROM t_int2
 WHERE a % 5 < 3 AND a::int + b::int < a::int * b::int;
SELECT a,b, a&b v1, a|b v2, null::int, a#b v3, ~a v4, a>>3 v5, b<<2 v6
  INTO pg_temp.test_s04a
  FROM t_int2
 WHERE a % 4 = 3 and a::int + b::int < a::int * b::int;
SELECT l.a, r.b, l.a::int + r.a::int v1, l.b::int - r.b::int v2
  INTO pg_temp.test_s05a
  FROM t_int1 l, t_int2 r
 WHERE l.a = r.a AND l.b < r.b;
SELECT l.a, null::int, r.b
  INTO pg_temp.test_s06a
  FROM t_int1 l, t_int2 r
 WHERE (l.a = r.a OR l.b = r.b)
   AND l.a % 100 in (37,53) AND r.b % 100 in (28,79);

SET pg_strom.enabled = off;
SELECT a, b
  INTO pg_temp.test_s01b
  FROM t_int2
 WHERE a > b AND @b > a;
SELECT a,b,-a v1, @b v2
  INTO pg_temp.test_s02b
  FROM t_int2
 WHERE (a < b AND @a > b) OR a = b;
SELECT a, b, a::int + b::int v1, a::int - b::int v2, a::int * b::int v3
  INTO pg_temp.test_s03b
  FROM t_int2
 WHERE a % 5 < 3 AND a::int + b::int < a::int * b::int;
SELECT a,b, a&b v1, a|b v2, null::int, a#b v3, ~a v4, a>>3 v5, b<<2 v6
  INTO pg_temp.test_s04b
  FROM t_int2
 WHERE a % 4 = 3 AND a::int + b::int < a::int * b::int;
SELECT l.a, r.b, l.a::int + r.a::int v1, l.b::int - r.b::int v2
  INTO pg_temp.test_s05b
  FROM t_int1 l, t_int2 r
 WHERE l.a = r.a AND l.b < r.b;
SELECT l.a, null::int, r.b
  INTO pg_temp.test_s06b
  FROM t_int1 l, t_int2 r
 WHERE (l.a = r.a OR l.b = r.b)
   AND l.a % 100 in (37,53) AND r.b % 100 in (28,79);

(SELECT * FROM pg_temp.test_s01a EXCEPT ALL SELECT * FROM pg_temp.test_s01b);
(SELECT * FROM pg_temp.test_s01b EXCEPT ALL SELECT * FROM pg_temp.test_s01a);
(SELECT * FROM pg_temp.test_s02a EXCEPT ALL SELECT * FROM pg_temp.test_s02b);
(SELECT * FROM pg_temp.test_s02b EXCEPT ALL SELECT * FROM pg_temp.test_s02a);
(SELECT * FROM pg_temp.test_s03a EXCEPT ALL SELECT * FROM pg_temp.test_s03b);
(SELECT * FROM pg_temp.test_s03b EXCEPT ALL SELECT * FROM pg_temp.test_s03a);
(SELECT * FROM pg_temp.test_s04a EXCEPT ALL SELECT * FROM pg_temp.test_s04b);
(SELECT * FROM pg_temp.test_s04b EXCEPT ALL SELECT * FROM pg_temp.test_s04a);
(SELECT * FROM pg_temp.test_s05a EXCEPT ALL SELECT * FROM pg_temp.test_s05b);
(SELECT * FROM pg_temp.test_s05b EXCEPT ALL SELECT * FROM pg_temp.test_s05a);
(SELECT * FROM pg_temp.test_s06a EXCEPT ALL SELECT * FROM pg_temp.test_s06b);
(SELECT * FROM pg_temp.test_s06b EXCEPT ALL SELECT * FROM pg_temp.test_s06a);

--
-- int4
--
RESET pg_strom.enabled;
SELECT c, d
  INTO pg_temp.test_i01a
  FROM t_int2
 WHERE c > d AND @d > c;
SELECT c, d, -c v1, @d v2
  INTO pg_temp.test_i02a
  FROM t_int2
 WHERE (c < d AND @c > d) OR c = d;
SELECT c, d, c::bigint + d v1, c - d::bigint v2, c::bigint * d::bigint v3
  INTO pg_temp.test_i03a
  FROM t_int2
 WHERE c % 5 < 3 AND d IS NOT NULL;
SELECT c&d v1, c|d v2, null::int, c#d v3, ~d v4, c>>3 v5, d<<2 v6
  INTO pg_temp.test_i04a
  FROM t_int2
 WHERE d % 4 = 3 AND c < d;
SELECT l.c lc, r.d, l.c::bigint + r.d::bigint v1, l.d::bigint - r.d::bigint v2
  INTO pg_temp.test_i05a
  FROM t_int1 l, t_int2 r
 WHERE l.c = r.d AND l.d < r.d;
SELECT l.c, null::int, r.d
  INTO pg_temp.test_i06a
  FROM t_int1 l, t_int2 r
 WHERE (l.c % 100000 = r.c % 100000 OR l.d % 400000 = r.d % 400000)
   AND l.c % 100 in (17,83) AND r.d % 100 in (31,57);

SET pg_strom.enabled = off;
SELECT c, d
  INTO pg_temp.test_i01b
  FROM t_int2
 WHERE c > d AND @d > c;
SELECT c, d, -c v1, @d v2
  INTO pg_temp.test_i02b
  FROM t_int2
 WHERE (c < d AND @c > d) OR c = d;
SELECT c, d, c::bigint + d v1, c - d::bigint v2, c::bigint * d::bigint v3
  INTO pg_temp.test_i03b
  FROM t_int2
 WHERE c % 5 < 3 AND d IS NOT NULL;
SELECT c&d v1, c|d v2, null::int, c#d v3, ~d v4, c>>3 v5, d<<2 v6
  INTO pg_temp.test_i04b
  FROM t_int2
 WHERE d % 4 = 3 AND c < d;
SELECT l.c lc, r.d, l.c::bigint + r.d::bigint v1, l.d::bigint - r.d::bigint v2
  INTO pg_temp.test_i05b
  FROM t_int1 l, t_int2 r
 WHERE l.c = r.d AND l.d < r.d;
SELECT l.c, null::int, r.d
  INTO pg_temp.test_i06b
  FROM t_int1 l, t_int2 r
 WHERE (l.c % 100000 = r.c % 100000 OR l.d % 400000 = r.d % 400000)
   AND l.c % 100 in (17,83) AND r.d % 100 in (31,57);

(SELECT * FROM pg_temp.test_i01a EXCEPT ALL SELECT * FROM pg_temp.test_i01b);
(SELECT * FROM pg_temp.test_i01b EXCEPT ALL SELECT * FROM pg_temp.test_i01a);
(SELECT * FROM pg_temp.test_i02a EXCEPT ALL SELECT * FROM pg_temp.test_i02b);
(SELECT * FROM pg_temp.test_i02b EXCEPT ALL SELECT * FROM pg_temp.test_i02a);
(SELECT * FROM pg_temp.test_i03a EXCEPT ALL SELECT * FROM pg_temp.test_i03b);
(SELECT * FROM pg_temp.test_i03b EXCEPT ALL SELECT * FROM pg_temp.test_i03a);
(SELECT * FROM pg_temp.test_i04a EXCEPT ALL SELECT * FROM pg_temp.test_i04b);
(SELECT * FROM pg_temp.test_i04b EXCEPT ALL SELECT * FROM pg_temp.test_i04a);
(SELECT * FROM pg_temp.test_i05a EXCEPT ALL SELECT * FROM pg_temp.test_i05b);
(SELECT * FROM pg_temp.test_i05b EXCEPT ALL SELECT * FROM pg_temp.test_i05a);
(SELECT * FROM pg_temp.test_i06a EXCEPT ALL SELECT * FROM pg_temp.test_i06b);
(SELECT * FROM pg_temp.test_i06b EXCEPT ALL SELECT * FROM pg_temp.test_i06a);

--
-- int8
--
RESET pg_strom.enabled;
SELECT e, f
  INTO pg_temp.test_l01a
  FROM t_int2
 WHERE e > f AND @e > f;
SELECT e, f, -e v1, @f v2
  INTO pg_temp.test_l02a
  FROM t_int2
 WHERE (e < f AND @e > f) OR e = f;
SELECT e, f, e + f v1, e - f v2, e * f v3
  INTO pg_temp.test_l03a
  FROM t_int2
 WHERE e % 5 < 3 AND f IS NOT NULL;
SELECT e&f v1, e|f v2, null::int, e#f v3, ~e v4, e>>3 v5, f<<2 v6
  INTO pg_temp.test_l04a
  FROM t_int2
 WHERE e % 4 = 3 AND e < f;
SELECT l.e, r.f, l.e + r.f v1, l.f - r.e v2
  INTO pg_temp.test_l05a
  FROM t_int1 l, t_int2 r
 WHERE l.e = r.e AND l.f < r.f;
SELECT l.e, null::int, r.f
  INTO pg_temp.test_l06a
  FROM t_int1 l, t_int2 r
 WHERE (l.e % 100000 = r.e % 100000 OR l.f % 400000 = r.f % 400000)
   AND l.e % 100 in (19,76) AND r.f % 100 in (29,61);

SET pg_strom.enabled = off;
SELECT e, f
  INTO pg_temp.test_l01b
  FROM t_int2
 WHERE e > f AND @e > f;
SELECT e, f, -e v1, @f v2
  INTO pg_temp.test_l02b
  FROM t_int2
 WHERE (e < f AND @e > f) OR e = f;
SELECT e, f, e + f v1, e - f v2, e * f v3
  INTO pg_temp.test_l03b
  FROM t_int2
 WHERE e % 5 < 3 AND f IS NOT NULL;
SELECT e&f v1, e|f v2, null::int, e#f v3, ~e v4, e>>3 v5, f<<2 v6
  INTO pg_temp.test_l04b
  FROM t_int2
 WHERE e % 4 = 3 AND e < f;
SELECT l.e, r.f, l.e + r.f v1, l.f - r.e v2
  INTO pg_temp.test_l05b
  FROM t_int1 l, t_int2 r
 WHERE l.e = r.e AND l.f < r.f;
SELECT l.e, null::int, r.f
  INTO pg_temp.test_l06b
  FROM t_int1 l, t_int2 r
 WHERE (l.e % 100000 = r.e % 100000 OR l.f % 400000 = r.f % 400000)
   AND l.e % 100 in (19,76) AND r.f % 100 in (29,61);

(SELECT * FROM pg_temp.test_l01a EXCEPT ALL SELECT * FROM pg_temp.test_l01b);
(SELECT * FROM pg_temp.test_l01b EXCEPT ALL SELECT * FROM pg_temp.test_l01a);
(SELECT * FROM pg_temp.test_l02a EXCEPT ALL SELECT * FROM pg_temp.test_l02b);
(SELECT * FROM pg_temp.test_l02b EXCEPT ALL SELECT * FROM pg_temp.test_l02a);
(SELECT * FROM pg_temp.test_l03a EXCEPT ALL SELECT * FROM pg_temp.test_l03b);
(SELECT * FROM pg_temp.test_l03b EXCEPT ALL SELECT * FROM pg_temp.test_l03a);
(SELECT * FROM pg_temp.test_l04a EXCEPT ALL SELECT * FROM pg_temp.test_l04b);
(SELECT * FROM pg_temp.test_l04b EXCEPT ALL SELECT * FROM pg_temp.test_l04a);
(SELECT * FROM pg_temp.test_l05a EXCEPT ALL SELECT * FROM pg_temp.test_l05b);
(SELECT * FROM pg_temp.test_l05b EXCEPT ALL SELECT * FROM pg_temp.test_l05a);
(SELECT * FROM pg_temp.test_l06a EXCEPT ALL SELECT * FROM pg_temp.test_l06b);
(SELECT * FROM pg_temp.test_l06b EXCEPT ALL SELECT * FROM pg_temp.test_l06a);
