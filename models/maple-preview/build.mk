# Model-owned explicit CPU build; private qualification fixtures are excluded.
.PHONY: maple-server
maple-server:
	bash models/maple-preview/build.sh
