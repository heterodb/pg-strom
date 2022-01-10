require "helper"
require "fluent/plugin/out_arrow_file.rb"

class ArrowFileOutputTest < Test::Unit::TestCase
  TMP_DIR = File.expand_path(File.dirname(__FILE__) + "/../tmp/out_file#{ENV['TEST_ENV_NUMBER']}")

  DEF_CONFIG = %[
    path #{TMP_DIR}/arrow_file_test.arrow
    schema_defs "uint8_column=Uint8"
  ]

  def setup
    Fluent::Test.setup
  end

  test "mytest" do
    d = create_driver %[
      path /tmp/test_path
      schema_defs "uint8_column=Uint8"
    ]
    assert_equal '/tmp/test_path',d.instance.path 
  end

  private

  def create_driver(conf)
    Fluent::Test::Driver::Output.new(Fluent::Plugin::ArrowFileOutput).configure(conf)
  end
end
