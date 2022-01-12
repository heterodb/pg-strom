require "helper"
require "fluent/plugin/out_arrow_file.rb"

class ArrowFileOutputTest < Test::Unit::TestCase
  # Define directory path where the test output file exists.
  TMP_DIR = File.expand_path(File.dirname(__FILE__) + "/../out_file#{ENV['TEST_ENV_NUMBER']}")

  DEFAULT_CONFIG = %[
    path #{TMP_DIR}/arrow_file_test.arrow
    schema_defs "uint8_column=Uint8"
  ]
  DEFALUT_TAG='test_tag'

  def setup
    Fluent::Test.setup
    FileUtils.mkdir_p TMP_DIR
  end

  sub_test_case 'configuration' do
    test "conf_paths" do
      d = create_driver
      assert_equal "#{TMP_DIR}/arrow_file_test.arrow",d.instance.path 
    end
  end
=begin
  test "feed_test" do
    d = create_driver
    d.run(default_tag: 'test_tag') do
      d.feed({'uint8_column' => 93})
    end

    assert system("../../arrow-tools/arrow2csv --header #{TMP_DIR}/arrow_file_test.arrow")
  end

  test "error_test" do
    d = create_driver
    assert_raise RangeError do
      d.run(default_tag: 'test_tag') do
        d.feed({'uint8_column' => 257})
      end
    end
  end  
=end

  sub_test_case 'data_type' do
    test "uint8_ok" do
      d = create_driver

      d.run(default_tag: DEFALUT_TAG,flush:true,shutdown:false) do
        d.feed({'uint8_column' => 0})
        d.feed({'uint8_column' => 255})
      end

      assert_raise do
        d.run(default_tag: DEFALUT_TAG,flush:true,shutdown:false) do
          d.feed({'uint8_column' => 256})
        end
      end
      
      system("../../arrow-tools/arrow2csv --header #{TMP_DIR}/arrow_file_test.arrow")
    end
  end
=begin
      # NG case: lower limit over
      assert_raises RangeError do 
        d.run(default_tag: 'test_tag') do
          d.feed({'uint8_column' => 1})
        end
      end
    end
  end

  test "error_test" do
    d = create_driver
    assert_raise RangeError do
      d.run(default_tag: 'test_tag') do
        d.feed({'uint8_column' => 257})
      end
    end
  end
=end

  teardown do
    FileUtils.rm_rf TMP_DIR
  end

  def create_driver(conf = DEFAULT_CONFIG,opts={})
    Fluent::Test::Driver::Output.new(Fluent::Plugin::ArrowFileOutput, opts: opts).configure(conf)
  end
end
