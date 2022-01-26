lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)

Gem::Specification.new do |spec|
  spec.name    = "fluent-plugin-arrow-file"
  spec.version = "0.2"
  spec.authors = ["KaiGai Kohei"]
  spec.email   = ["kaigai@heterodb.com"]

  spec.summary       = %q{Fluentd output plugin for Apache Arrow files.}
  spec.description   = %q{Fluentd output plugin for Apache Arrow files. See documentation for more details.}
  spec.homepage      = "https://github.com/heterodb/pg-strom/"
  spec.license       = "PostgreSQL"
  spec.extensions    = %w[ext/arrow_file_write/extconf.rb]

  test_files, files  = "Gemfile Makefile \
                        fluent-plugin-arrow-file.gemspec \
                        lib/fluent/plugin/out_arrow_file.rb \
                        ext/arrow_file_write/arrow_defs.h \
                        ext/arrow_file_write/arrow_ipc.h \
                        ext/arrow_file_write/float2.h \
                        ext/arrow_file_write/arrow_ruby.c \
                        ext/arrow_file_write/arrow_nodes.c \
                        ext/arrow_file_write/arrow_write.c \
                        ext/arrow_file_write/extconf.rb".split(" ").partition do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.files         = files
  spec.executables   = files.grep(%r{^bin/}) { |f| File.basename(f) }
  spec.test_files    = test_files
  spec.require_paths = ["lib"]

  spec.add_development_dependency "bundler", "~> 2.2.32"
  spec.add_development_dependency "rake", "~> 13.0.6"
  spec.add_development_dependency "test-unit", "~> 3.3.4"
  spec.add_runtime_dependency "fluentd", [">= 0.14.10", "< 2"]
end
