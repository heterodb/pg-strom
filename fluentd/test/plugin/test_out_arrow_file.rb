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

    assert system("../arrow-tools/arrow2csv --header #{TMP_DIR}/arrow_file_test.arrow")
  end

    test "uint8" do

      uint8_conf = %[
        path #{TMP_DIR}/uint8_test.arrow
        schema_defs "uint8_column=Uint8"
      ]

      assert_nothing_raised do
        feed_record(uint8_conf,{'uint8_column' => 0})
        feed_record(uint8_conf,{'uint8_column' => 255})
      end

      assert_raise RangeError do
        feed_record(uint8_conf,{'uint8_column' => 256})
      end
      assert_raise RangeError do
        feed_record(uint8_conf,{'uint8_column' => -1})
      end
    end

  test "error_test" do
    d = create_driver
    assert_raise RangeError do
      d.run(default_tag: 'test_tag') do
        d.feed({'uint8_column' => 257})
      end
    endassert system("../arrow-tools/arrow2csv --header #{TMP_DIR}/arrow_file_test.arrow")
  end  2013-02-28 12:34:56.789
=end

  def feed_record(conf,record)
    d = create_driver(conf)
    d.run(default_tag: DEFALUT_TAG,flush:true,shutdown:true) do
      d.feed(record)
    end
  end

  sub_test_case 'data_type' do

=begin
    test "timestamp" do
      conf = %[
        path #{TMP_DIR}/timestamp_ns_test.arrow
        schema_defs "tsns=Timestamp"
      ]
      d=create_driver(conf)

      ## EventTimeが何故かIntegerになってします。

      #t1=event_time("2016-10-03 23:58:09 UTC")
      #record={'nonpo' => t1}
      #p record.class

      #p record['tsns'].class
      
      d.run(default_tag: DEFAULT_CONFIG) do
        ## FLuent::EventTime
        #d.feed({'tsns' => 5})
        #d.feed({'tsns' => event_time("2016-10-03 23:58:09 UTC")})
        d.feed({'tsns' => '2013-02-28 12:34:56.789012'})
        #d.feed('rururur',t1,record)
      end

      assert system("../arrow-tools/arrow2csv --header #{TMP_DIR}/timestamp_ns_test.arrow")
    end
=end

    test "uint8" do
      conf = %[
        path #{TMP_DIR}/uint8_test.arrow
        schema_defs "ui8=Uint8"
      ]

      d=create_driver(conf)

      d.run(default_tag: DEFALUT_TAG) do
        d.feed({'ui8' => 0})
        d.feed({'ui8' => 255})
      end

      # TODO: 比較チェックする。
      assert system("../arrow-tools/arrow2csv --header #{TMP_DIR}/uint8_test.arrow")
    end


    test "float64" do
      conf = %[
        path #{TMP_DIR}/float_64_test.arrow
        schema_defs "f64=Float64"
      ]

      d=create_driver(conf)

      d.run(default_tag: DEFALUT_TAG) do
        d.feed({'f64' => 1.0009765625})
        d.feed({'f64' => 3.1415926535})
        d.feed({'f64' => -3.14159})
        d.feed({'f64' => 0.1})
      end

      # TODO: 比較チェックする。
      assert system("../arrow-tools/arrow2csv --header #{TMP_DIR}/float_64_test.arrow")
    end


    test "float32" do
      conf = %[
        path #{TMP_DIR}/float_32_test.arrow
        schema_defs "f32=Float32"
      ]

      d=create_driver(conf)

      d.run(default_tag: DEFALUT_TAG) do
        d.feed({'f32' => 1.0009765625})
        d.feed({'f32' => 0.1})
        d.feed({'f32' => 3.1415926535})
        d.feed({'f32' => -3.14159})
      end

      # TODO: 比較チェックする。
      assert system("../arrow-tools/arrow2csv --header #{TMP_DIR}/float_32_test.arrow")
    end

    test "float16" do
      conf = %[
        path #{TMP_DIR}/float_16_test.arrow
        schema_defs "f16=Float16"
      ]

      d=create_driver(conf)

      d.run(default_tag: DEFALUT_TAG) do
        d.feed({'f16' => 1.0009765625})
        d.feed({'f16' => 3.1415926535})
        d.feed({'f16' => 0.1})
        d.feed({'f16' => -3.14159})
      end

      # TODO: 比較チェックする。
      assert system("../arrow-tools/arrow2csv --header #{TMP_DIR}/float_16_test.arrow")
    end

    test "decimal1" do
      conf = %[
        path #{TMP_DIR}/decimal_1.arrow
        schema_defs "dec1=Decimal"
      ]

      d=create_driver(conf)

      d.run(default_tag: DEFALUT_TAG) do
        d.feed({'dec1' => 3.141})     # ???
        d.feed({'dec1' => 2.71828})   # ???
      end

      assert system("../arrow-tools/arrow2csv --header #{TMP_DIR}/decimal_1.arrow")
    end

    test "bool" do
      conf = %[
        path #{TMP_DIR}/bool.arrow
        schema_defs "bl1=Bool,tsns=Int64"
        ts_column "tsns"
      ]

      d=create_driver(conf)

      d.run(default_tag: DEFALUT_TAG) do
        d.feed({'bl1' => true})     # ???
        d.feed({'bl1' => false})   # ???
        d.feed({'bl1' => nil})   # ???
      end

      assert system("../arrow-tools/arrow2csv --header #{TMP_DIR}/bool.arrow")
    end

    test "utf8" do
      conf = %[
        path #{TMP_DIR}/utf8_1.arrow
        schema_defs "utf81=Utf8"
      ]

      d=create_driver(conf)

      d.run(default_tag: DEFALUT_TAG) do
        d.feed({'utf81' => "fuga"})     # ???
        d.feed({'utf81' => "ほげ彅"})   # ???
        d.feed({'utf81' => nil})   # ???
      end

      assert system("../arrow-tools/arrow2csv --header #{TMP_DIR}/utf8_1.arrow")
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
