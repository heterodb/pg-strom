require "helper"
require "fluent/plugin/out_arrow_file.rb"

# テスト実行方法
# cd ~/pg-strom/fluentd/fluent-plugin-arrow-file
# bundle exec rake test

class ArrowFileOutputTest < Test::Unit::TestCase
  TMP_DIR = File.expand_path(File.dirname(__FILE__) + "/../tmp/out_file#{ENV['TEST_ENV_NUMBER']}")

  DEFAULT_CONFIG = %[
    path #{TMP_DIR}/arrow_file_test.arrow
    schema_defs "uint8_column=Uint8"
  ]

  def setup
    Fluent::Test.setup
    FileUtils.mkdir_p TMP_DIR
  end

  test "conf_paths" do
    d = create_driver
    assert_equal "#{TMP_DIR}/arrow_file_test.arrow",d.instance.path 
  end

  test "feed_test" do
    d = create_driver
    d.run(default_tag: 'test_tag') do
      d.feed({'uint8_column' => 93})
    end

    assert system("../../arrow-tools/arrow2csv --header #{TMP_DIR}/arrow_file_test.arrow")
  end

  teardown do
    FileUtils.rm_rf TMP_DIR
  end


  def create_driver(conf = DEFAULT_CONFIG,opts={})
    Fluent::Test::Driver::Output.new(Fluent::Plugin::ArrowFileOutput, opts: opts).configure(conf)
  end
end
