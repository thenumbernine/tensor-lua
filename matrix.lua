--[[

    File: matrix.lua

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



require 'tensor.delta'

-- TODO rename, maybe merge with Tensor

-- returns the determinant of a NxN matrix
-- = leviCivita(N){a1,...,aN} * m{a1,1} * m{a2,2} * ... * m{aN,N}
function det(m)
	assert(m.rank == 2)
	assert(m.dim[1] == m.dim[2])
	local dim = m.dim[1]
	local eps = leviCivita(dim)
	local sig = {}
	for i=1,dim do
		sig[i] = 'i'..i
	end
	local expr = eps:rep(sig)
	for i=1,dim do
		expr = expr * m:rep{'i'..i, i}
	end
	return -expr:assign()
end

function inv(m)
	assert(m.rank == 2)
	assert(m.dim[1] == m.dim[2])
	local dim = m.dim[1]
	local detValue = det(m)
	if detValue == 0 then return false, "cannot compute inverse of a singular matrix" end
	local d = kroneckerDelta(dim)
	local dis = table()
	for i=1,dim*2 do
		dis:insert('i'..i)
	end
	local expr = d:rep(dis)
	for i=2,dim do
		expr = expr * m{'i'..(dim+i), 'i'..i}
	end
	local res = expr:assign{'i'..1, 'i'..(dim+1)}
	res = res * (-1/(3*2 * detValue))
	return res
end


