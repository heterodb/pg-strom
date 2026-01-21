RUBY_RAKE=/opt/fluent/bin/rake
ifeq ($(shell test -x $(RUBY_RAKE) || echo -n 0),0)
RUBY_RAKE=rake
endif

test:
	@echo "Running tests from parent directory..."
	@cd .. && $(RUBY_RAKE) test 2>&1 | grep -v -E "(literal string will be frozen|warning: assigned but unused variable)"

.PHONY: test
