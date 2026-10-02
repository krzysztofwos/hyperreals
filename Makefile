.PHONY: all lean lean-audit
all:
	$(MAKE) -C paper

lean:
	lake build

lean-audit:
	./scripts/check-lean-trust.sh
