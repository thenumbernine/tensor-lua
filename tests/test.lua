--[[

    File: test.lua 

    Copyright (C) 2000-2013 Christopher Moore (christopher.e.moore@gmail.com)
	  
    This software is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
  
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
  
    You should have received a copy of the GNU General Public License along
    with this program; if not, write the Free Software Foundation, Inc., 51
    Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

--]]


require 'tensor'
require 'tensor.notebook'

notebook[[
a = tensor.Tensor(4,4,function(i,j) return j+4*(i-1) end)
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