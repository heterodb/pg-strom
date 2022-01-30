require "fluent/plugin/output"
require "arrow_file_write"

module Fluent
  module Plugin
    class ArrowFileOutput < Fluent::Plugin::Output
      Fluent::Plugin.register_output("arrow_file", self)

      helpers :inject, :compat_parameters

      desc "The Path of the arrow file"
      config_param :path, :string
      config_param :schema_defs, :string
      config_param :ts_column, :string, default: NIL
      config_param :tag_column, :string, default: NIL
      config_param :filesize_threshold, :integer, default: 10000

      config_section :buffer do
        config_set_default :@type, 'memory'
        config_set_default :chunk_keys, ['tag']
        # fit pg2arrow default 256MB
        config_set_default :chunk_limit_size, 256 * 1024 * 1024
      end

      def prefer_buffered_processing
        true
      end
  
      def multi_workers_ready?
        false
      end

      def configure(conf)
        compat_parameters_convert(conf, :buffer, :inject, default_chunk_key: "time")
        super

        @af=ArrowFileWrite.new(@path,@schema_defs,{"ts_column" => @ts_column,"tag_column" => @tag_column,"filesize_threshold" => @filesize_threshold})
      end

      def format(tag,time,record)
        r = inject_values_to_record(tag, time, record)
        [tag,time,r].to_msgpack
      end

      def formatted_to_msgpack_binary?
        true
      end

      def write(chunk)
        @af.writeChunk(chunk)
      end
    end
  end
end
