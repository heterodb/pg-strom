#!/usr/bin/env python3
"""
Data generation script for Parquet and Apache Arrow formats
Generates sample data with various data types and outputs to Parquet or Arrow files.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.ipc as ipc
import numpy as np
import random
import string
from datetime import datetime, date, time, timedelta
from decimal import Decimal
import os

# Define output target columns
# Modify this list to control which columns are included in the output file
OUTPUT_TARGET_COLUMNS = [
    "a_int32",
    "b_int64", 
    "c_int96",
    "d_float",
    "e_double",
    "f_string",
    "g_enum",
    "h_decimal",
    "i_date",
    # "j_time",  # Commented out due to Arrow FDW bug
    "k_timestamp"
]

# Define output format: 'parquet', 'arrow', 'csv', 'both', 'all'
# 'both' = parquet + arrow, 'all' = parquet + arrow + csv
OUTPUT_FORMAT = 'all'

def generate_random_data(num_records=100):
    """
    Generate random data for specified data types
    
    Args:
        num_records (int): Number of records to generate
        
    Returns:
        dict: Dictionary containing fields for each data type
    """
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    data = {}
    
    # INT32: 32-bit signed integer
    data['a_int32'] = np.random.randint(-2147483648, 2147483647, size=num_records, dtype=np.int32)
    
    # INT64: 64-bit signed integer
    data['b_int64'] = np.random.randint(-9223372036854775808, 9223372036854775807, size=num_records, dtype=np.int64)
    
    # INT96: For timestamp use (INT96 is deprecated, so using TIMESTAMP instead)
    base_time = datetime(2020, 1, 1)
    data['c_int96'] = [base_time + timedelta(days=random.randint(0, 1000)) for _ in range(num_records)]
    
    # FLOAT: 32-bit floating point number
    data['d_float'] = np.random.uniform(-1e10, 1e10, size=num_records).astype(np.float32)
    
    # DOUBLE: 64-bit floating point number
    data['e_double'] = np.random.uniform(-1e100, 1e100, size=num_records).astype(np.float64)
    
    # STRING: UTF-8 encoded string
    def generate_random_string(length=10):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    data['f_string'] = [generate_random_string(random.randint(5, 20)) for _ in range(num_records)]
    
    # ENUM: Enumeration type (implemented as string)
    enum_values = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'PURPLE', 'ORANGE']
    data['g_enum'] = [random.choice(enum_values) for _ in range(num_records)]
    
    # DECIMAL: Arbitrary precision decimal number
    decimal_values = []
    for _ in range(num_records):
        # Generate decimal with precision=10, scale=2
        integer_part = random.randint(0, 99999999)  # 8-digit integer part
        decimal_part = random.randint(0, 99)        # 2-digit decimal part
        decimal_value = Decimal(f"{integer_part}.{decimal_part:02d}")
        decimal_values.append(decimal_value)
    data['h_decimal'] = decimal_values
    
    # DATE: Date
    start_date = date(2020, 1, 1)
    data['i_date'] = [start_date + timedelta(days=random.randint(0, 1000)) for _ in range(num_records)]
    
    # TIME: Time (microsecond precision)
    time_values = []
    for _ in range(num_records):
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        microsecond = random.randint(0, 999999)
        time_values.append(time(hour, minute, second, microsecond))
    data['j_time'] = time_values
    
    # TIMESTAMP: Timestamp (microsecond precision)
    base_timestamp = datetime(2020, 1, 1)
    data['k_timestamp'] = [
        base_timestamp + timedelta(
            days=random.randint(0, 1000),
            seconds=random.randint(0, 86399),
            microseconds=random.randint(0, 999999)
        ) for _ in range(num_records)
    ]
    
    return data

def create_data_schema(target_columns):
    """
    Define schema for data file based on target columns
    
    Args:
        target_columns (list): List of column names to include
        
    Returns:
        pyarrow.Schema: Defined schema
    """
    # Define all available column schemas
    all_schemas = {
        'a_int32': pa.int32(),
        'b_int64': pa.int64(),
        'c_int96': pa.timestamp('us'),  # Using timestamp instead of INT96
        'd_float': pa.float32(),
        'e_double': pa.float64(),
        'f_string': pa.string(),
        'g_enum': pa.string(),  # ENUM saved as string
        'h_decimal': pa.decimal128(10, 2),  # precision=10, scale=2
        'i_date': pa.date32(),
        'j_time': pa.time64('us'),  # microsecond precision
        'k_timestamp': pa.timestamp('us')  # microsecond precision
    }
    
    # Create schema with only target columns
    schema_fields = [(col, all_schemas[col]) for col in target_columns if col in all_schemas]
    schema = pa.schema(schema_fields)
    
    return schema

def convert_to_pyarrow_table(data, schema, target_columns):
    """
    Convert dictionary data to PyArrow table based on target columns
    
    Args:
        data (dict): Data dictionary
        schema (pyarrow.Schema): Schema definition
        target_columns (list): List of column names to include
        
    Returns:
        pyarrow.Table: Converted table
    """
    # Define type mapping for PyArrow arrays
    type_mapping = {
        'a_int32': pa.int32(),
        'b_int64': pa.int64(),
        'c_int96': pa.timestamp('us'),
        'd_float': pa.float32(),
        'e_double': pa.float64(),
        'f_string': pa.string(),
        'g_enum': pa.string(),
        'h_decimal': pa.decimal128(10, 2),  # Use Decimal objects directly
        'i_date': pa.date32(),
        'j_time': pa.time64('us'),
        'k_timestamp': pa.timestamp('us')
    }
    
    # Create PyArrow arrays for target columns only
    arrays = []
    for col in target_columns:
        if col in data and col in type_mapping:
            arrays.append(pa.array(data[col], type=type_mapping[col]))
    
    # Create table
    table = pa.Table.from_arrays(arrays, schema=schema)
    return table

def save_to_parquet(table, filename='python_generated_data.parquet'):
    """
    Save PyArrow table to Parquet file
    
    Args:
        table (pyarrow.Table): Table to save
        filename (str): Output filename
    """
    output_path = os.path.join(os.path.dirname(__file__), filename)
    
    # Write to Parquet file
    pq.write_table(table, output_path, compression='snappy')
    
    print(f"Parquet file created successfully: {output_path}")
    
    # Display file information
    file_size = os.path.getsize(output_path)
    print(f"File size: {file_size:,} bytes")

def save_to_arrow(table, filename='python_generated_data.arrow'):
    """
    Save PyArrow table to Apache Arrow format file
    
    Args:
        table (pyarrow.Table): Table to save
        filename (str): Output filename
    """
    output_path = os.path.join(os.path.dirname(__file__), filename)
    
    # Write to Arrow file using Arrow IPC format
    with pa.OSFile(output_path, 'wb') as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    
    print(f"Arrow file created successfully: {output_path}")
    
    # Display file information
    file_size = os.path.getsize(output_path)
    print(f"File size: {file_size:,} bytes")

def save_to_csv(table, filename='python_generated_data.csv'):
    """
    Save PyArrow table to CSV file
    
    Args:
        table (pyarrow.Table): Table to save
        filename (str): Output filename
    """
    output_path = os.path.join(os.path.dirname(__file__), filename)
    
    # Convert to Pandas DataFrame and save as CSV
    df = table.to_pandas()
    df.to_csv(output_path, index=False)
    
    print(f"CSV file created successfully: {output_path}")
    
    # Display file information
    file_size = os.path.getsize(output_path)
    print(f"File size: {file_size:,} bytes")

def save_table(table, base_filename='python_generated_data', output_format='both'):
    """
    Save PyArrow table to specified format(s)
    
    Args:
        table (pyarrow.Table): Table to save
        base_filename (str): Base filename without extension
        output_format (str): 'parquet', 'arrow', 'csv', 'both', or 'all'
    """
    if output_format in ['parquet', 'both']:
        save_to_parquet(table, f'{base_filename}.parquet')
    
    if output_format in ['arrow', 'both']:
        save_to_arrow(table, f'{base_filename}.arrow')
    
    if output_format == 'csv':
        save_to_csv(table, f'{base_filename}.csv')
    
    if output_format == 'all':
        save_to_parquet(table, f'{base_filename}.parquet')
        save_to_arrow(table, f'{base_filename}.arrow')
        save_to_csv(table, f'{base_filename}.csv')

def display_sample_data(table, num_rows=5):
    """
    Display sample of generated data
    
    Args:
        table (pyarrow.Table): Table to display
        num_rows (int): Number of rows to display
    """
    print(f"\n=== Sample Data (First {num_rows} rows) ===")
    
    # Convert to Pandas DataFrame for display
    df = table.to_pandas()
    print(df.head(num_rows))
    
    print(f"\n=== Data Type Information ===")
    print(df.dtypes)
    
    print(f"\n=== Data Statistics ===")
    print(f"Total records: {len(df):,}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Included columns: {list(df.columns)}")

def main():
    """Main processing"""
    print("=== Data Generation Tool (Parquet & Apache Arrow) ===")
    print("Generates random data with various data types and outputs to specified format(s).\n")
    
    # Display configuration
    print(f"Target columns: {OUTPUT_TARGET_COLUMNS}")
    print(f"Number of target columns: {len(OUTPUT_TARGET_COLUMNS)}")
    print(f"Output format: {OUTPUT_FORMAT}\n")
    
    # Generate data
    print("1. Generating random data...")
    data = generate_random_data(num_records=100)
    
    # Define schema
    print("2. Defining data schema...")
    schema = create_data_schema(OUTPUT_TARGET_COLUMNS)
    
    # Convert to PyArrow table
    print("3. Converting to PyArrow table...")
    table = convert_to_pyarrow_table(data, schema, OUTPUT_TARGET_COLUMNS)
    
    # Save to specified format(s)
    print("4. Saving to file(s)...")
    save_table(table, 'python_generated_data', OUTPUT_FORMAT)
    
    # Display sample data
    display_sample_data(table)
    
    print("\n=== Processing Complete ===")

def validate_files():
    """
    Validate created files by reading them back
    """
    print("\n=== File Validation ===")
    
    # Check if files exist and read them
    arrow_file = 'python_generated_data.arrow'
    csv_file = 'python_generated_data.csv'
    
    if os.path.exists(arrow_file):
        print(f"Reading {arrow_file}...")
        try:
            with pa.OSFile(arrow_file, 'rb') as source:
                with ipc.open_file(source) as reader:
                    table = reader.read_all()
                    print(f"  - Successfully read {table.num_rows} rows, {table.num_columns} columns")
                    print(f"  - Schema: {table.schema}")
        except Exception as e:
            print(f"  - Error reading {arrow_file}: {e}")
    
    if os.path.exists(csv_file):
        print(f"Reading {csv_file}...")
        try:
            df = pd.read_csv(csv_file)
            print(f"  - Successfully read {len(df)} rows, {len(df.columns)} columns")
            print(f"  - Columns: {list(df.columns)}")
        except Exception as e:
            print(f"  - Error reading {csv_file}: {e}")

if __name__ == "__main__":
    main()
    
    # Validate files if they were created
    if OUTPUT_FORMAT in ['arrow', 'csv', 'both', 'all']:
        validate_files()
