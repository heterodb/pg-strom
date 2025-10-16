# Python Data Generator for PG-Strom Testing

This directory contains Python scripts for generating test data for PG-Strom testing purposes.

## Overview

`parquet_gen.py` is a script that generates test data with various data types to test PG-Strom's Arrow FDW functionality. It can output data in the following formats:

- **Parquet format** (.parquet)
- **Apache Arrow format** (.arrow) 
- **CSV format** (.csv)

The generated data includes diverse data types such as integers, floating-point numbers, strings, date/time, and decimal types.

## Setup

### 1. Create and activate virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage Example

### Basic execution

```bash
python parquet_gen.py
```

By default, three files are generated: `python_generated_data.parquet`, `python_generated_data.arrow`, and `python_generated_data.csv`.

### Sample output

```
=== Data Generation Tool (Parquet & Apache Arrow) ===
Generates random data with various data types and outputs to specified format(s).

Target columns: ['a_int32', 'b_int64', 'c_int96', 'd_float', 'e_double', 'f_string', 'g_enum', 'h_decimal', 'i_date', 'k_timestamp']
Number of target columns: 10
Output format: all

1. Generating random data...
2. Defining data schema...
3. Converting to PyArrow table...
4. Saving to file(s)...
Parquet file created successfully: /path/to/python_generated_data.parquet
File size: 10,344 bytes
Arrow file created successfully: /path/to/python_generated_data.arrow
File size: 10,482 bytes
CSV file created successfully: /path/to/python_generated_data.csv
File size: 14,917 bytes

=== Sample Data (First 5 rows) ===
      a_int32              b_int64    c_int96       d_float      e_double          f_string  g_enum    h_decimal      i_date                k_timestamp
0  -538846106  8662307466024843665 2021-10-16  8.165318e+09 -4.111022e+99  mq6OLkTkx9NIQ0Wo  PURPLE   9023909.30  2022-02-17 2021-08-09 19:32:21.157131
1  1273642419  5075304826311743019 2020-04-24 -5.208762e+09 -2.298045e+99            Xye4JS    BLUE  86592295.46  2021-01-14 2020-06-17 11:56:47.895993
...

=== Processing Complete ===
```

## Generated Data Types

- `a_int32`: 32-bit integer
- `b_int64`: 64-bit integer  
- `c_int96`: Timestamp
- `d_float`: 32-bit floating-point number
- `e_double`: 64-bit floating-point number
- `f_string`: String
- `g_enum`: Enumeration type (implemented as string)
- `h_decimal`: Fixed-point decimal number (precision=10, scale=2)
- `i_date`: Date
- `k_timestamp`: Timestamp (microsecond precision)

## Testing Purpose

The generated files are used in regression tests to verify that PG-Strom's Arrow FDW can read Parquet and Arrow files correctly, and that their contents match the CSV file.

## Notes

- TIME type (`j_time`) is commented out due to an Arrow FDW bug
- Data is generated using a fixed seed, so identical data is produced on each execution