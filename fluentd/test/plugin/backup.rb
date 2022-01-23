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