require 'mkmf'
$CFLAGS='-D_GNU_SOURCE'
create_makefile('arrow_file_write')
