# Reference: https://stackoverflow.com/a/17845120

SUBDIRS := $(wildcard */.)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
.PHONY: all $(SUBDIRS) clean
