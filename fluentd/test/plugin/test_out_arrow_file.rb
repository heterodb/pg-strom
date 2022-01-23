require "helper"
require "fluent/plugin/out_arrow_file.rb"

class ArrowFileOutputTest < Test::Unit::TestCase
  TMP_DIR = File.expand_path(File.dirname(__FILE__) + "/../out_file#{ENV['TEST_ENV_NUMBER']}")
  EXPECTED_DIR = File.expand_path(File.dirname(__FILE__) + "/../expected")
  COMPARE_CMD=File.expand_path(File.dirname(__FILE__) + "/../compare_result.sh")

  class << self
    # Define directory path where the test output file exists.
    def startup
      p "create #{TMP_DIR}"
      FileUtils.rm_rf TMP_DIR
      FileUtils.mkdir_p TMP_DIR
      FileUtils.rm_f File.expand_path(File.dirname(__FILE__) + "/../regression.diff")
    end
  end

  DEFAULT_CONFIG = %[
    path #{TMP_DIR}/arrow_file_test.arrow
    schema_defs "uint8_column=Uint8"
  ]
  DEFALUT_TAG='test_tag'

  sub_test_case 'data_type' do
    test "uint_test" do
      conf = %[
        path #{TMP_DIR}/uint8_test.arrow
        schema_defs "ui1=Uint8,ui2=Uint16,ui3=Uint32,ui4=Uint64"
      ]

      d=create_driver(conf)

      d.run(default_tag: DEFALUT_TAG) do
        d.feed({'ui1' => 0,'ui2' => 0, 'ui3' => 0, 'ui4' => 0})
        d.feed({'ui1' => 255,'ui2' => 32767, 'ui3' => 2147483647, 'ui4' => 4294967295})
        d.feed({'ui1' => nil,'ui2' => nil, 'ui3' => nil, 'ui4' => nil})
      end

      assert system("#{COMPARE_CMD} #{TMP_DIR}/uint8_test.arrow #{EXPECTED_DIR}/uint8_test.out")
    end

    test "timestamp_check" do
      conf = %[
        path #{TMP_DIR}/timestamp_ns_test.arrow
        schema_defs "tsns1=Timestamp,tsns2=Timestamp[sec],tsns3=Timestamp[ms],tsns4=Timestamp[us],tsns5=Timestamp[ns]"  #" #
      ]
      d=create_driver(conf)

      t1=event_time("2016-10-03 23:58:09 UTC")
      time_string='2000-02-29 12:34:56.789012'

      assert_nothing_raised do
        d.run(default_tag: DEFAULT_CONFIG) do
          d.feed({'tsns1' => t1, 'tsns2' => t1, 'tsns3' => t1, 'tsns4' => t1, 'tsns5' => t1})
          d.feed({'tsns1' => time_string,'tsns2' => time_string, 'tsns3' => time_string, 'tsns4' => time_string, 'tsns5' => time_string})
          d.feed({'tsns1' => nil,'tsns2' => nil, 'tsns3' => nil,'tsns4' => nil, 'tsns5' => nil})
        end
      end
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
        #d.feed({'dec1' => 3.141592})     # ???
        #d.feed({'dec1' => 2.436})   # ???
        d.feed({'dec1' => 123})
        d.feed({'dec1' => 456})
        d.feed({'dec1' => 789})
        d.feed({'dec1' => 987})
        d.feed({'dec1' => 654})
        d.feed({'dec1' => 321})
        #d.feed({'dec1' => 0.1})   # ???
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

    test "compare_expected" do
      assert system("ls -lah #{TMP_DIR}")
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

  def create_driver(conf = DEFAULT_CONFIG,opts={})
    Fluent::Test::Driver::Output.new(Fluent::Plugin::ArrowFileOutput, opts: opts).configure(conf)
  end
end
