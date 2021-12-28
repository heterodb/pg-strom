
require "fluent/plugin/output"
require_relative '/home/onishi/pg-strom/fluentd/ArrowFile.so'

module Fluent
  module Plugin
    class ArrowFileOutput < Fluent::Plugin::Output
      Fluent::Plugin.register_output("arrow_file", self)

      helpers :inject, :compat_parameters

      desc "The Path of the arrow file"
      config_param :path, :string
      config_param :schema_defs, :string

      config_section :buffer do
        config_set_default :@type, 'memory'
        config_set_default :chunk_keys, ['tag']
        config_set_default :chunk_limit_size, 2 * 1024 * 1024
      end

      def prefer_buffered_processing
        true
      end
  
      def multi_workers_ready?
        false
      end

      def configure(conf)
        compat_parameters_convert(conf, :buffer, :inject, default_chunk_key: "time")

        p conf
        super

        @af=ArrowFile.new(@path,@schema_defs,{"ts_column" => "yourtime","tag_column" => "yourtag","filesize_threshold" => 10000})
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
