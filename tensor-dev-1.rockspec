package = "tensor"
version = "dev-1"
source = {
	url = "git+https://github.com/thenumbernine/tensor-lua"
}
description = {
	summary = "numerical tensor index notation library",
	detailed = "numerical tensor index notation library",
	homepage = "https://github.com/thenumbernine/tensor-lua",
	license = "MIT"
}
dependencies = {
	"lua ~> 5.1"
}
build = {
	type = "builtin",
	modules = {
		["tensor.delta"] = "delta.lua",
		["tensor.index"] = "index.lua",
		["tensor.layer"] = "layer.lua",
		["tensor.matrix"] = "matrix.lua",
		["tensor.notebook"] = "notebook.lua",
		["tensor.representation"] = "representation.lua",
		["tensor"] = "tensor.lua",
		["tensor.tests.delta"] = "tests/delta.lua",
		["tensor.tests.inverse"] = "tests/inverse.lua",
		["tensor.tests.metric"] = "tests/metric.lua",
		["tensor.tests.test"] = "tests/test.lua"
	},
}
