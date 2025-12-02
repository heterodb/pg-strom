/*
 * arrow_ruby.c
 *
 * A Ruby language extension to write out data as Apache Arrow files.
 * --
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <ruby.h>






// ----------------------------------------------------------------
//
// Interface for Ruby ABI
//
// ----------------------------------------------------------------
extern "C" {
	static VALUE rb_ArrowFileWrite__initialize(VALUE self,
											   VALUE __pathname,
											   VALUE __schema_defs,
											   VALUE __params);
	static VALUE rb_ArrowFileWrite__writeChunk(VALUE self,
											   VALUE chunk);
	void	Init_arrow_file_write(void);
};

static VALUE
rb_ArrowFileWrite__initialize(VALUE self,
							  VALUE __pathname,
							  VALUE __schema_defs,
							  VALUE __params)
{
	rb_require("time");

	return self;
}

static VALUE
rb_ArrowFileWrite__writeChunk(VALUE self,
							  VALUE chunk)
{
	VALUE	retval = 0;
	

	return retval;
}

void
Init_arrow_file_write(void)
{
	VALUE	klass;

	klass = rb_define_class("ArrowFileWrite",  rb_cObject);
	rb_define_method(klass, "initialize", rb_ArrowFileWrite__initialize, 3);
	rb_define_method(klass, "writeChunk", rb_ArrowFileWrite__writeChunk, 1);
}
