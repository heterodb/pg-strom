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
      FileUtils.rm_f File.expand_path(File.dirname(__FILE__) + "/../regression.diffs")
    end
  end

  DEFAULT_CONFIG = %[
    path #{TMP_DIR}/arrow_file_test.arrow
    schema_defs "uint8_column=Uint8"
  ]
  DEFALUT_TAG='test_tag'

  sub_test_case 'data_type' do
    def get_driver(file_name,schema_defs)
      conf = %[
        path #{TMP_DIR}/#{file_name}.arrow
        schema_defs "#{schema_defs}"
      ]

      return create_driver(conf)
    end

    def compare_arrow(file_name)
      system("#{COMPARE_CMD} #{TMP_DIR}/#{file_name}.arrow #{EXPECTED_DIR}/#{file_name}.out -s")
    end

    test "uint_test" do
      file_name='uint_test'
      d=get_driver(file_name,"ui1=Uint8,ui2=Uint16,ui3=Uint32,ui4=Uint64")
      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'ui1' => 0,'ui2' => 0, 'ui3' => 0, 'ui4' => 0})
          d.feed({'ui1' => 255,'ui2' => 32767, 'ui3' => 2147483647, 'ui4' => 4294967295})
          d.feed({'ui1' => nil,'ui2' => nil, 'ui3' => nil, 'ui4' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "timestamp_check" do
      file_name='timestamp_check'
      d=get_driver(file_name,"tsns1=Timestamp,tsns2=Timestamp[sec],tsns3=Timestamp[ms],tsns4=Timestamp[us],tsns5=Timestamp[ns]")

      t1=event_time("2016-10-03 23:58:09 UTC")
      time_string='2000-02-29 12:34:56.789012'

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'tsns1' => t1, 'tsns2' => t1, 'tsns3' => t1, 'tsns4' => t1, 'tsns5' => t1})
          d.feed({'tsns1' => time_string,'tsns2' => time_string, 'tsns3' => time_string, 'tsns4' => time_string, 'tsns5' => time_string})
          d.feed({'tsns1' => nil,'tsns2' => nil, 'tsns3' => nil,'tsns4' => nil, 'tsns5' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "float_check" do
      file_name='float_check'
      d=get_driver(file_name,"float1=Float16,float2=Float32,float3=Float64")

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'float1' => 3.96875, 'float2' => 3.96875, 'float3' => 3.96875})
          d.feed({'float1' => -1.984375, 'float2' => -1.984375, 'float3' => -1.984375})
          d.feed({'float1' => nil, 'float2' => nil, 'float3' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    # decimal skipping...
    test "bool_check" do
      file_name='bool_check'
      d=get_driver(file_name,"bool1=Bool")

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'bool1' => true})
          d.feed({'bool1' => false})
          d.feed({'bool1' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "utf8_check" do
      file_name='utf8_check'
      d=get_driver(file_name,"utf81=Utf8")

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'utf81' => "fuga"})
          d.feed({'utf81' => "ほげ彅😀"})   # text including special charactors.
          d.feed({'utf81' => "ほげほげ".encode("EUC-JP")})    # text encoded in EUC-JP
          d.feed({'utf81' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "ip_check" do
      file_name='ip_check'
      d=get_driver(file_name,"ip1=Ipaddr4")

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'ip1' => IPAddr.new("192.168.0.1").to_s})
          #d.feed({'ip1' => "192.168.0.1/24",'ip2' => nil})
        end
      end
      assert compare_arrow(file_name)
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
