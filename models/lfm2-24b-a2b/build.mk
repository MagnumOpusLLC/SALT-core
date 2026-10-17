# Model-owned explicit CPU build; private qualification fixtures are excluded.
.PHONY: lfm-server
lfm-server:
	bash models/lfm2-24b-a2b/build.sh
