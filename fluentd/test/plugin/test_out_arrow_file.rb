require "helper"
require "fluent/plugin/out_arrow_file.rb"
require 'date'

class ArrowFileOutputTest < Test::Unit::TestCase
  TMP_DIR = File.expand_path(File.dirname(__FILE__) + "/../result")
  EXPECTED_DIR = File.expand_path(File.dirname(__FILE__) + "/../expected")
  COMPARE_CMD=File.expand_path(File.dirname(__FILE__) + "/../compare_result.sh")
  COMPARE_METADATA_CMD=File.expand_path(File.dirname(__FILE__) + "/../compare_metadata.sh")
  ARROW2CSV_CMD=File.expand_path(File.dirname(__FILE__) + "/../../../arrow-tools/arrow2csv")
  GET_ROW_NUM_CMD=File.expand_path(File.dirname(__FILE__) + "/../get_arrows_rows.sh")

  # Common 
  class << self
    # Define directory path where the test output file exists.
    def startup
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

  def compare_arrow(file_name)
    system("#{COMPARE_CMD} #{TMP_DIR}/#{file_name}.arrow #{EXPECTED_DIR}/#{file_name}.out")
  end

  def get_driver(file_name,schema_defs)
    conf = %[
      path #{TMP_DIR}/#{file_name}.arrow
      schema_defs "#{schema_defs}"
    ]

    return create_driver(conf)
  end

  # Data type check
  ## http://heterodb.github.io/pg-strom/fluentd/#configuration

    test "uint_test" do
      file_name='uint_test'
      d=get_driver(file_name,"ui1=Uint8,ui2=Uint16,ui3=Uint32,ui4=Uint64")
      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'ui1' => 0,'ui2' => 0, 'ui3' => 0, 'ui4' => 0})
          d.feed({'ui1' => 255,'ui2' => 32767, 'ui3' => 2147483647, 'ui4' => 4294967295})
          d.feed({'ui1' => "255",'ui2' => "32767", 'ui3' => "2147483647", 'ui4' => "4294967295"})
          d.feed({'ui1' => nil,'ui2' => nil, 'ui3' => nil, 'ui4' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "int_test" do
      file_name='int_test'
      d=get_driver(file_name,"i1=Int8,i2=Int16,i3=Int32,i4=Int64")
      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'i1' => -128,'i2' => -32767, 'i3' => -2147483647, 'i4' => -4294967295})
          d.feed({'i1' => 127,'i2' => 32767, 'i3' => 2147483647, 'i4' => 4294967295})
          d.feed({'i1' => "127",'i2' => "32767", 'i3' => "2147483647", 'i4' => "4294967295"})
          d.feed({'i1' => nil,'i2' => nil, 'i3' => nil, 'i4' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "timestamp_test" do
      file_name='timestamp_test'
      d=get_driver(file_name,"tsns1=Timestamp,tsns2=Timestamp[sec],tsns3=Timestamp[ms],tsns4=Timestamp[us],tsns5=Timestamp[ns]")

      t1=event_time("2016-10-03 23:58:09 UTC")
      time_string='2000-02-29 12:34:56.789012'
      time_string_with_timezone='2000-02-29 12:34:56.789012 JST'
      time_string_with_timediff='2000-02-29 12:34:56.789012+0500'

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'tsns1' => t1, 'tsns2' => t1, 'tsns3' => t1, 'tsns4' => t1, 'tsns5' => t1})
          d.feed({'tsns1' => time_string,'tsns2' => time_string, 'tsns3' => time_string, 'tsns4' => time_string, 'tsns5' => time_string})
          d.feed({'tsns1' => time_string_with_timezone,'tsns2' => time_string_with_timezone, 'tsns3' => time_string_with_timezone, 'tsns4' => time_string_with_timezone, 'tsns5' => time_string_with_timezone})
          d.feed({'tsns1' => time_string_with_timediff,'tsns2' => time_string_with_timediff, 'tsns3' => time_string_with_timediff, 'tsns4' => time_string_with_timediff, 'tsns5' => time_string_with_timediff})
          d.feed({'tsns1' => nil,'tsns2' => nil, 'tsns3' => nil,'tsns4' => nil, 'tsns5' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "date_test" do
      file_name='date_test'
      d=get_driver(file_name,"dns1=Date,dns2=Date[ms],dns3=Date[day]")

      t1=event_time("2016-10-03 23:58:09 UTC")
      time_string='2000-02-29 12:34:56.789012'

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'dns1' => t1, 'dns2' => t1, 'dns3' => t1})
          d.feed({'dns1' => "2016-10-03",'dns2'=>"2016-10-03",'dns3'=>"2016-10-03"})
          d.feed({'dns1' => time_string,'dns2' => time_string, 'dns3' => time_string})
          d.feed({'dns1' => nil,'dns2' => nil, 'dns3' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "time_test" do
      file_name='time_test'
      d=get_driver(file_name,"tsns1=Time,tsns2=Time[ns],tsns3=Time[us],tsns4=Time[ms],tsns5=Time[sec]")

      t1=event_time("2016-10-03 23:58:09 UTC")
      time_string='2000-02-29 12:34:56.789012 UTC'

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'tsns1' => t1, 'tsns2' => t1, 'tsns3' => t1, 'tsns4' => t1, 'tsns5' => t1})
          d.feed({'tsns1' => time_string,'tsns2' => time_string, 'tsns3' => time_string, 'tsns4' => time_string, 'tsns5' => time_string})
          d.feed({'tsns1' => nil,'tsns2' => nil, 'tsns3' => nil,'tsns4' => nil, 'tsns5' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "float_test" do
      file_name='float_test'
      d=get_driver(file_name,"float1=Float16,float2=Float32,float3=Float64")

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'float1' => 3.96875, 'float2' => 3.96875, 'float3' => 3.96875})
          d.feed({'float1' => -1.984375, 'float2' => -1.984375, 'float3' => -1.984375})
          d.feed({'float1' => "-1.984375", 'float2' => "-1.984375", 'float3' => "-1.984375"})
          d.feed({'float1' => nil, 'float2' => nil, 'float3' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "decimal_test" do
      file_name='decimal_test'
      d=get_driver(file_name,"d1=Decimal,d2=Decimal128,d3=Decimal(38)")

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'d1' => 0,'d2' => 0,'d3' => 0})
          # Float value.
          d.feed({'d1' => 12345678901234567890.12345678,'d2' => 12345678901234567890.12345678,'d3'=>1.12345678901234567890123456789012345678})
          # string
          d.feed({'d1' => "12345678901234567890.12345678",'d2' => "12345678901234567890.12345678",'d3'=>"1.12345678901234567890123456789012345678"})
          d.feed({'d1' => nil,'d2' => nil,'d3' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "bool_test" do
      file_name='bool_test'
      d=get_driver(file_name,"bool1=Bool")

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'bool1' => true})
          d.feed({'bool1' => false})
          d.feed({'bool1' => "true"})
          d.feed({'bool1' => "false"})
          d.feed({'bool1' => "TRUE"})
          d.feed({'bool1' => "FALSE"})
          d.feed({'bool1' => "True"})
          d.feed({'bool1' => "False"})
          d.feed({'bool1' => "t"})
          d.feed({'bool1' => "f"})
          d.feed({'bool1' => "T"})
          d.feed({'bool1' => "F"})
          d.feed({'bool1' => "1"})
          d.feed({'bool1' => "0"})
          d.feed({'bool1' => 1})
          d.feed({'bool1' => 0})
          d.feed({'bool1' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "utf8_test" do
      file_name='utf8_test'
      d=get_driver(file_name,"utf81=Utf8")

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'utf81' => "fuga0123456789"})
          d.feed({'utf81' => "ほ げ禰彅a　0😀"})   # text including special charactors.
          d.feed({'utf81' => "ほげほげ".encode("EUC-JP")})    # text encoded in EUC-JP
          d.feed({'utf81' => nil})
        end
      end
      assert compare_arrow(file_name)
    end

    test "ip_test" do
      file_name='ip_test'
      d=get_driver(file_name,"ip1=Ipaddr4,ip2=Ipaddr6")

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'ip1' => "192.168.0.1",'ip2' => "b085:fe52:e3c1:bc49:5fab:65de:64d8:d5b8"})
          d.feed({'ip1' => "0.0.0.0",'ip2' => "::"})
          d.feed({'ip1' => "0.0.0.1",'ip2' => "::1"})
          d.feed({'ip1' => "255.255.255.255",'ip2' => "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"})
          d.feed({'ip1' => nil,'ip2' => nil})
        end
      end
      assert compare_arrow(file_name)
    end
  
  # Configuration Check
  ## Refer: http://heterodb.github.io/pg-strom/fluentd/#configuration
    test "timedate_file_test" do
      conf =%[
        path #{TMP_DIR}/test_%Y_%y_%m_%d_%H_%M_%S_%p.arrow
        schema_defs "dc=Utf8"
      ]
      d = create_driver(conf)
      ct=Time.new
      pid=Process.pid
      correct_filename="test_#{ct.year}_#{ct.year%100}_#{format("%02d",ct.month)}_#{format("%02d",ct.day)}_#{format("%02d",ct.hour)}_#{format("%02d",ct.min)}_#{format("%02d",ct.sec)}_#{pid}.arrow"
      correct_filepath="#{TMP_DIR}/#{correct_filename}"

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed({'dc' => "success"})
        end
      end
      p "Checking path: #{correct_filepath}"
      # Just checking the file exists.(output is ignored.)
      assert system("#{ARROW2CSV_CMD} #{correct_filepath}")
    end

    test "column_replace_test" do
      file_name="column_replace_test"
      conf =%[
        path #{TMP_DIR}/#{file_name}.arrow
        schema_defs "new_time=Timestamp,new_tag=Utf8,payload=Utf8"
        ts_column "new_time"
        tag_column "new_tag"
      ]
      d = create_driver(conf)

      t1=event_time("2000-02-29 23:59:09 UTC")
      t2=event_time("1999-03-18 01:23:45 UTC")
      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          d.feed('replaced_tag',t1, {'payload' => "success"})
          # new_time and new_tag are also in record, and this should be replaced.
          d.feed('replaced_tag',t1, {'new_time' => t2,'payload' => "success"})
          d.feed('replaced_tag',t1, {'new_tag' => 'old_tag','payload' => "success"})
          d.feed('replaced_tag',t1, {'new_time' => t2, 'new_tag' => 'old_tag','payload' => "success"})
        end
      end
      assert compare_arrow(file_name)
    end

    test "switching_files" do
      file_path="#{TMP_DIR}/switch_test_%H_%M_%S.arrow"
      generate_row_num=100
      conf =%[
        path #{file_path}
        schema_defs "num=Uint32"

        <buffer>
          chunk_limit_records 10
        </buffer>
      ]
      d = create_driver(conf)

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          for i in 1..generate_row_num do
            d.feed({'num' => i})
            sleep(0.01)
          end
        end
      end
      # getting sum of the number of rows in generated arrow files, and it should equals generate_row_num
      assert `#{GET_ROW_NUM_CMD} '#{TMP_DIR}/switch_test*'`.to_s.to_i == generate_row_num 
    end

    test "filesize_threshold_test" do
      file_path="#{TMP_DIR}/threshold.arrow"
      generate_row_num=8192
      payload_size=4096
      conf =%[
        path #{file_path}
        schema_defs "payload=Utf8"
        filesize_threshold 16

        <buffer>
          chunk_limit_size 3MB
        </buffer>
      ]
      d = create_driver(conf)

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          generate_row_num.times do
            # generate random text to create heavy file.
            txt=(0...payload_size).map { (65 + rand(26)).chr }.join
            d.feed({'payload' => txt})
          end
        end
      end
      # getting sum of the number of rows in generated arrow files, and it should equals generate_row_num
      assert `#{GET_ROW_NUM_CMD} '#{file_path}*'`.to_s.to_i == generate_row_num 
    end

    test "statistics_test" do
      file_name="statistics_test"
      generate_row_num=255

      conf =%[
        path #{TMP_DIR}/#{file_name}.arrow
        schema_defs "num1=Uint8;stat_enabled,num2=Uint16;stat_enabled,num3=Uint32;stat_enabled,num4=Uint64;stat_enabled,
        num5=Int8;stat_enabled,num6=Int16;stat_enabled,num7=Int32;stat_enabled,num8=Int64;stat_enabled,
        num9=Float16;stat_enabled,num10=Float32;stat_enabled,num11=Float64;stat_enabled,
        num12=Decimal;stat_enabled,ts1=Timestamp;stat_enabled,ts2=Date;stat_enabled,ts3=Time;stat_enabled,
        bool1=Bool,string1=Utf8,ip41=Ipaddr4,ip61=Ipaddr6"   # checking whether these types can be inserted without errors, or not; in situation: many columns, many rows.

        <buffer>
          chunk_limit_records 100
        </buffer>
      ]

      d = create_driver(conf)

      t1=DateTime.new(2000,2,28,12,34,56)

      assert_nothing_raised do
        d.run(default_tag: DEFALUT_TAG) do
          for i in 1..generate_row_num do
            ts_text=t1.next_day(i*0.1-12).to_s
            d.feed({'num1' => i,'num2' => i*100,'num3' => i*10000000,'num4' => i*10000000,
              'num5' => i-128,'num6' => (i*255)-32767,'num7' => (i*16777215)-2147483647,'num8' => (i*33554431)-4294967295,
              'num9' => -2.0 + i*0.01,'num10' => -0.02 + i * 0.001,'num11' => -0.0002 + i * 0.00001,
              'num12' => i*100000,
              'ts1' => ts_text, 'ts2' => ts_text,'ts3' => ts_text,
              'bool1' => i%2==0,
              'string1' => 'hello', 'ip41' => '192.168.0.1', 'ip61' => 'b085:fe52:e3c1:bc49:5fab:65de:64d8:d5b8'})
          end
        end
      end
      # getting sum of the number of rows in generated arrow files, and check it equals generate_row_num
      system("#{COMPARE_METADATA_CMD} #{TMP_DIR}/#{file_name}.arrow #{EXPECTED_DIR}/#{file_name}.dump")
    end

  def create_driver(conf = DEFAULT_CONFIG,opts={})
    Fluent::Test::Driver::Output.new(Fluent::Plugin::ArrowFileOutput, opts: opts).configure(conf)
  end
end
