# How to run plugin test

## Install test tools

Preparation:
* http://heterodb.github.io/pg-strom/fluentd/#installation
* arrow2csv( Run `make arrow2csv` in `pg-strom/arrow-tools`)

```bash
cd pg-strom/fluentd
bundle
```

## Run all test cases

```
cd pg-strom/fluentd
rake test
```

## Run one test case

```bash
rake test TESTOPTS="-n'test: <test name>'"
```