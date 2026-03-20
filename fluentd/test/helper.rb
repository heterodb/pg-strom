$LOAD_PATH.unshift(File.expand_path("../../", __FILE__))

# NOTE: Suppress warnings about frozen string literals from Fluentd gem
#
# This warning originates from Fluentd 1.19.1 gem's internal code:
#   /opt/fluent/lib/ruby/gems/3.4.0/gems/fluentd-1.19.1/lib/fluent/plugin/buffer/memory_chunk.rb:25
#   @chunk = ''.force_encoding(Encoding::ASCII_8BIT)
#
# This is a bug in Fluentd gem itself, not in our codebase.
# The warning will be resolved when Fluentd updates to support Ruby 3.4's frozen string literals.
# Until then, we suppress these warnings to keep test output clean.
#
# Reference: In Ruby 3.4+, string literals will be frozen by default.
# The Fluentd gem should use String.new('') or +''.force_encoding(...) instead.
Warning[:deprecated] = false if Warning.respond_to?(:[]=)

require "test-unit"
require "fluent/test"
require "fluent/test/driver/output"
require "fluent/test/helpers"

Test::Unit::TestCase.include(Fluent::Test::Helpers)
Test::Unit::TestCase.extend(Fluent::Test::Helpers)
