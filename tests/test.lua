#!/usr/bin/env lua
local notebook = require 'tensor.notebook'
-- global for notebook
Tensor = require 'tensor'

notebook[[
a = Tensor(4,4,function(i,j) return j+4*(i-1) end)
=a
=a'ij'
=a'ij''ij'
=a'ij''ji'
=a'ij''i'
=a'ij''j'
=a'ij'''

=a'ii'
=a'ii''i'
=a'ii'''	-- trace

=(a'ik' * a'kj')'ij'	-- mat square

=(a'ik' * a'jk')'ij'	-- dots

-- raise/lower with lorentz metric

=a'ij'
=a'_i^j'
=a'^i_j'
=a'_ij'

=a'ij''_ij'
=a'_i^j''_ij'
=a'^i_j''_ij'
=a'_ij''_ij'

]]
