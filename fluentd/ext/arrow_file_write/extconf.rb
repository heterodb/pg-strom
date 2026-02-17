require 'mkmf'
pkg_config("arrow")    # -larrow
pkg_config("parquet")  # -lparquet
#$CPPFLAGS << " -g -O0 "
create_makefile('arrow_file_write')
