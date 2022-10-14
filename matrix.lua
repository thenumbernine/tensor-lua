local kroneckerDelta = require 'tensor.delta'
local table = require 'ext.table'

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

return {
	det = det,
	inv = inv,
}
