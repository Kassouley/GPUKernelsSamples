BASE_DIR := ./src/hip

SUBDIRS := $(shell find $(BASE_DIR) -type f -name 'Makefile' -exec dirname {} \;)

all: $(SUBDIRS)

$(SUBDIRS):
	@$(MAKE) -C $@

clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
