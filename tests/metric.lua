require 'symmath'
require 'tensor'
require 'tensor.notebook'

-- [[ polar
r = symmath.Variable('r')
phi = symmath.Variable('phi')
coords = {r, phi}
dim = #coords
x = tensor.Tensor(dim, {r * symmath.cos(phi), r * symmath.sin(phi)})
--]]

--[[ spherical
r = symmath.Variable('r')
theta = symmath.Variable('theta')
phi = symmath.Variable('phi')
coords = {r, theta, phi}
dim = #coords
x = tensor.Tensor(dim, {
	r * symmath.cos(phi) * symmath.sin(theta),
	r * symmath.sin(phi) * symmath.sin(theta),
	r * symmath.cos(theta)
})
--]]

-- [[ coordinate basis
local coordBasis = setmetatable(table.map(coords, function(x)
	return function(y)
		return symmath.diff(y, x)
	end
end), nil)
--]]
--[[ non-coordinate basis
local coordBasis = {
	function(y) return symmath.diff(y, r) end,
	function(y) return symmath.diff(y, phi) * (1/r) end,
}
--]]
tensor.Tensor:setCoordinateBasis(unpack(coordBasis))

print('x = '..x)

-- if we want a comma derivative then we need,
-- more than just a metric, a coordinate basis

local e = table()
for i,coord in ipairs(coords) do
	e[i] = coordBasis[i](x)
	print('e_'..coords[i].name..' = '..e[i])
end


do
	local a = tensor.Tensor(dim, function(i)
		local v = symmath.Variable('a'..i)
		v.deferDiff = true
		return v
	end)
	for i,coordi in ipairs(coords) do
		for j, coordj in ipairs(coords) do
			print('[e_'..coordi.name..',e_'..coordj.name..'] = '..
				coordBasis[i](coordBasis[j](a)) - coordBasis[j](coordBasis[i](a))
			)
			
			-- now reproject them into the coordinate frame (the e's)
			-- and each of those is the c_ijk connection coefficients
		end
	end
end

-- if E is a matrix with column vectors the basis vectors e_* then E * E^T = g
g = tensor.Tensor(dim,dim,function(i,j)
	local ei = assert(e[i])
	local ej = assert(e[j])
	return (ei'i' * ej'i')''
end)
print('g = '..g)

do return end




-- assume the metric is diagonal, so inverting it is simply inverting the diagonal components...
gInv = tensor.Tensor(dim,dim,function(i,j)
	if i ~= j then
		assert(g:elem(i,j) == symmath.Constant(0))
		return 0
	else
		return 1 / g:elem(i,j)
	end
end)

--[[
-- here's an issue ...
-- tensors are specified in covariant terms
-- however metrics are typically specified in contravariant terms ...
-- sooo ... for now, swap the metric and its inverse
--]]
g, gInv = gInv, g

--[[
the metric inverse is used to transform from lower to upper during assignment
the elements specified are covariant (they abide by the transformation laws of the Tensor class)
--]]
tensor.Tensor:setMetricInverse(gInv)

--[[
now g has to have the the inverse elements of gInv
however it will appear inverted of what its elements should be (g^ij will be g_ij)
--]]
tensor.Tensor:setMetric(g)

--[[
so how do we perform operations on the metric?
--]]
print('g^ij = '..g'^ij')
print('g_ij = '..g'_ij')	

--[[
for polar, g_phi_phi_,r = 2*r, all else is zero
so g^phi_phi_,r = g^phi^alpha * g_alpha_phi_,r = g^phi^phi * g_phi_phi_,r = r^-2 * 2*r = 2*r^-1
so g^phi^phi_,r = g^phi^alpha * g^phi^alpha_,r = g^phi^phi * g^phi_phi,_r = r^-2 * 2*r^-1 = 2*r^-3
so g^phi^phi^,r = g^r^alpha * g^phi^phi_,alpha = g^r^r * g^phi^phi_,r = 1 * 2*r^-3 = 2*r^-3
--]]
print('g_ij,k = '..g'_ij,k')
print('g^ij,k = '..g'ij,k')

gamma = (.5 * (g'_ij,k' + g'_ik,j' - g'_jk,i'))'_ijk'
print('gamma_ijk',gamma'_ijk')
print('gamma^i_jk',gamma'^i_jk')

-- now for an attempt at the covariant derivative
-- (comma derivatives aren't associative with index gymnastics in a non-constant basis)
-- (but, due to the fact that g_ij;k = 0, we know that the covariant derivative does)
function covariantDerivative(t)
	local indexes = {}
	for i=1,t.rank do
		table.insert(indexes, 'i'..i)
	end
	table.insert(indexes, ',_x')
	
	local result = t:rep(indexes)
	
	table.remove(indexes)	-- remove the comma term from the end of our indexes
	for i=1,t.rank do
		indexes[i] = 'u'
		result = result + gamma{'i'..i, '_u', '_x'} * t:rep(indexes)
		indexes[i] = 'i'..i
	end
	
	table.insert(indexes, 'x')
	
	local r = result:assign(indexes)
	
	return r
end

--[[
a^i_;x = a^i_,x + gamma^i_u_x * a^u
for polar basis ...
let a^r = r, a^phi = r^2
a^r_,r = 1
a^r_,phi = 0
a^phi_,r = 2*r
a^phi_,phi = 0
gamma^r_u_r * a^u = gamma^r_r_r * a^r + gamma^r_phi_r * a^phi = 0 * a^r + 0 * a^phi = 0
gamma^r_u_phi * a^u = gamma^r_r_phi * a^r + gamma^r_phi_phi * a^phi = 0 * a^r + (-r) * a^phi = -r * r^2 = -r^3
gamma^phi_u_r * a^u = gamma^phi_r_r * a^r + gamma^phi_phi_r * a^phi = 0 * a^r + 1/r * a^phi = 1/r * r^2 = r
gamma^phi_u_phi * a^u = gamma^phi_r_phi * a^r + gamma^phi_phi_phi * a^phi = 1/r * a^r + 0 * a^phi = r / r = 1
a^r_;r = a^r_,r + gamma^r_u_r * a^u = 1 + 0 = 1
a^r_;phi = a^r_,phi + gamma^r_u_phi * a^u = 0 + -r^3 = -r^3
a^phi_;r = a^phi_,r + gamma^phi_u_r * a^u = 2*r + r = 3*r
a^phi_;phi = a^phi_,phi + gamma^phi_u_phi * a^u = 0 + 1 = 1
--]]
a = tensor.Tensor(dim, function(i) return r^i end)
print('a = '..a)
print('a^i_,x = '..a'i_,x''ix')
print('a^i_;x = '..covariantDerivative(a))


print('g_ij;k = '..covariantDerivative(g))

--[=[
non-coordinate basis:
[e_r,e_r] = d/dr[d/dr[x]] - d/dr[d/dr[x]] = 0
[e_r,e_phi]
	= d/dr[1/r * d/dphi[x]] - 1/r * d/dphi[d/dr[x]]
	= -1/r^2 * d/dphi[x] + 1/r * d/dr,phi[x] - 1/r d/dr,phi[x]
	= -1/r^2 * d/dphi[x]
[e_phi,e_r] = 1/r^2 * d/dphi[x]
[e_phi,e_phi] = 1/r * d/dphi[1/r * d/dphi[x]] = 1/r * d/dphi[1/r * d/dphi[x]] = 0

[e_r,e_phi] = {
	(
		(d/d{phi,r}[a1] * (r ^ -1))
		+
		(-1 * (r ^ -1) * d/d{phi,r}[a1])

		+
		(-1 * d/d{phi}[a1] * (r ^ -2))
		+
		((r ^ -2) * d/d{phi}[a1])
	)
	,
	((-1 * d/d{phi}[a2] * (r ^ -2)) + (d/d{phi,r}[a2] * (r ^ -1)) + ((r ^ -2) * d/d{phi}[a2]) + (-1 * (r ^ -1) * d/d{phi,r}[a2]))
}

--]=]
