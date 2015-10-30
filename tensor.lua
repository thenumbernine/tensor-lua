--[[

    File: tensor.lua

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

--[[
TODO rewrite this for better usage ...
put 'TensorLayer' in its own class
use 'Tensor' as the main class
do like symmath's tensors do -- store upper/lower information
don't transform by metric if the metric is identity
--]]


local class = require 'ext.class'
local table = require 'ext.table'

local TensorLayer = require 'tensor.layer'
local TensorIndex = require 'tensor.index'

local Tensor = class(TensorLayer)

--[[
t = Tensor(dim1, dim2, ..., dimN)
t = Tensor{dim1, dim2, ..., dimN}
t = Tensor{dim1, dim2, ..., dimN, values=values}
t = Tensor(dim1, dim2, ..., dimN, values)

Variables dim1 through dimN must be 1-based integers.

Values can either be a nested table of numbers
(i.e. {x,y,z} for rank-1, {{a,b,c},{d,e,f},{g,h,i}} for rank-2, etc.)
 or a function that accepts (i1, ..., iN) for indexes i1 through iN.

Currently indexes are only 1-based.

Accessing values of tensor elements is intuitive: t[i1][i2]...[iN]

Multiplication requires extra information

a_ik * b_kj results in a rank-3 tensor i,j,k

a:rep('ik') * b:rep('kj')

c_ij = a_ik * b_kj is a multiplication then a contraction along the 'k' index	

--]]
function Tensor:init(...)
	local args = {...}	-- support Tensor(mu,nu,...)
	
	local values
	if type(args[1]) == 'table' then
		if type(args[2]) ~= 'number' then
			values = args[2]	-- support Tensor({mu, nu, ...}, values)
		end
		args = args[1]	-- support Tensor{mu,nu,...}
	end

	if args.values then
		values = args.values	-- support Tensor{mu, nu, ..., values=...}
	end
	
	if type(args[#args]) ~= 'number' then
		values = table.remove(args)	-- support Tensor(mu, nu, ..., values)
	end
	
	self.rank = #args
	assert(self.rank > 0, "I don't support rank-0 tensors.  You need to provide at least one dimension.  If you were trying to use a true 0-rank tensor, just use a scalar.")

	self.dim = {}
	for i=1,self.rank do
		self.dim[i] = args[i]
	end

	self:fillRank(self, 1, values, {})
end

function Tensor:fillRank(t, dimIndex, values, indexes)
	t.layerDim = assert(self.dim[dimIndex], "failed to find dimIndex "..dimIndex)
	for i=1,t.layerDim do
		indexes[dimIndex] = i
		if dimIndex == #self.dim then	-- last dim, fill layer with zeroes
			local v
			if values then
				if type(values) == 'table' then
					v = values[i]
				elseif type(values) == 'function' then
					v = values(table.unpack(indexes))
				else
					error("don't know how to deduce the values you passed us")
				end
			end
			t[i] = v or 0
		else
			t[i] = TensorLayer()
			local rankValues
			if values then
				if type(values) == 'table' then
					rankValues = values[i]
				elseif type(values) == 'function' then
					rankValues = values
				else
					error("don't know how to deduce the values you passed us")
				end
			end
			self:fillRank(t[i], dimIndex+1, rankValues, indexes)
		end
	end
end

function Tensor:iterFunc(index)
	for i=self.rank,1,-1 do
		index[i] = index[i] + 1
		if index[i] <= self.dim[i] then break end
		index[i] = 1
		if i == 1 then return nil end
	end
	return index
end

--[[
returns an iterator that iterates across all elements of the tensor
the index will be in a table so that the number of indexes isn't fixed

t = Tensor(i,j)
for indexes in t:iter() do
	print(table.unpack(indexes))
end
will produce
1	1
1	2
...
1	N
2	1
...
M	1
...
M	N

--]]
function Tensor:iter()
	local index = {}
	for i=1,self.rank do
		index[i] = 1
	end
	index[self.rank] = 0	-- so the first increment will set it to 1,..,1
	return Tensor.iterFunc, self, index
end

function Tensor.__eq(a,b)
	local ta = getmetatable(a) == Tensor
	local tb = getmetatable(b) == Tensor
	if not ta or not tb then return false end		-- I don't support rank-0 tensors, so comparing a scalar means it's not equal
	if a.rank ~= b.rank then return false end
	for i=1,a.rank do
		if a.dim[i] ~= b.dim[i] then return false end
	end
	local function coIter(t)
		return coroutine.wrap(function()
			for is in t:iter() do
				coroutine.yield(t:elem(is))
			end
		end)
	end
	local ca = coIter(a)
	local cb = coIter(b)
	while true do
		local va = ca()
		local vb = cb()
		if va == nil then return true end	-- and assert vb == nil as well, so long as the dimensions match, as we tested above
		if va ~= vb then return false end
	end
end

-- outer product
-- c[i1..iM,j1..jN] = a[i1..iM] * b[j1..jN]
function Tensor.__mul(a,b)
	local ta = getmetatable(a) == Tensor
	local tb = getmetatable(b) == Tensor
	if ta and tb then
		return Tensor(table():append(a.dim):append(b.dim):append{function(...)
			local args = {...}
			assert(#args == #a.dim + #b.dim)
			local i = {}
			for n=1,#a.dim do
				i[n] = args[n]
			end
			local j = {}
			for n=1,#b.dim do
				j[n] = args[#a.dim + n]
			end
			return a:elem(i) * b:elem(j)
		end})
	elseif ta then
		return Tensor(a.dim, function(...)
			return a:elem(...) * b
		end)
	elseif tb then
		return Tensor(b.dim, function(...)
			return a * b:elem(...)
		end)
	end
end

function Tensor.__add(a,b)
	assert(a.rank == b.rank, "cannot add without matching rank")
	for i=1,a.rank do
		assert(a.dim[i] == b.dim[i], "cannot add without matching dimensions")
	end
	return Tensor(a.dim, function(...)
		return a:elem(...) + b:elem(...)
	end)
end

function Tensor.__sub(a,b)
	assert(a.rank == b.rank, "cannot subtract without matching rank")
	for i=1,a.rank do
		assert(a.dim[i] == b.dim[i], "cannot subtract without matching dimensions")
	end
	return Tensor(a.dim, function(...)
		return a:elem(...) - b:elem(...)
	end)
end

function Tensor.__div(a,b)
	return Tensor(a.dim, function(...)
		return a:elem(...) / b
	end)
end

function Tensor.__unm(a)
	return Tensor(a.dim, function(...)
		return -a:elem(...)
	end)
end

-- symmath extension
-- TODO how about a map function?
function Tensor:diff(...)
	local diffVars = {...}
	return Tensor(self.dim, function(...)
		return symmath.diff(self:elem(...), table.unpack(diffVars))
	end)
end
function Tensor:prune()
	return Tensor(self.dim, function(...)
		return symmath.prune(self:elem(...))
	end)
end

-- TODO how about a general function apply?

function Tensor:clone()
	return Tensor(self.dim, function(...)
		return self:elem(...)
	end)
end

--[[
produce a trace between dimensions i and j
store the result in dimension i, removing dimension j
--]]
function Tensor:trace(i,j)
	if i == j then
		error("cannot apply contraction across the same index: "..i)
	end

	if self.dim[i] ~= self.dim[j] then
		error("tried to apply tensor contraction across indices of differing dimension: "..i.."th and "..j.."th of "..table.concat(self.dim, ','))
	end
	
	local newdim = {table.unpack(self.dim)}
	-- remove the second index from the new dimension
	local removedDim = table.remove(newdim,j)
	-- keep track of where the first index is in the new dimension
	local newdimI = i
	if j < i then newdimI = newdimI - 1 end
	
	return Tensor(newdim, function(...)
		local indexes = {...}
		-- now when we reference the unremoved dimension

		local srcIndexes = {table.unpack(indexes)}
		table.insert(srcIndexes, j, indexes[newdimI])
		
		return self:elem(srcIndexes)
	end)
end

--[[
for all permutations of indexes other than i,
take each vector composed of index i
transform it by the provided rank-2 tensor
and store it back where you got it from
--]]
function Tensor:transformIndex(ti, m)
	assert(m.rank == 2, "can only transform an index by a rank-2 metric, got a rank "..m.rank)
	assert(m.dim[1] == m.dim[2], "can only transform an index by a square metric, got dims "..table.concat(m.dim,','))
	assert(self.dim[ti] == m.dim[1], "tried to transform tensor of dims "..table.concat(self.dim,',').." with metric of dims "..table.concat(m.dim,','))
	return Tensor(self.dim, function(...)
		-- current element being transformed
		local is = {...}
		local vxi = is[ti]	-- the current coordinate along the vector being transformed
		
		local result = 0
		for vi=1,m.dim[1] do
			local vis = {table.unpack(is)}
			vis[ti] = vi
			result = result + m:elem(vxi, vi) * self:elem(vis)
		end
		
		return result
	end)
end

--[[
technically this will incur, for M lowers set to true, DIM^M additions and (DIM-1)^M multiplies
if we were to use transformIndex then we'd only incur M*DIM additions and M*(DIM-1) multiplies,
but we'd need M*DIM^N memory (for rank-N), and as many assignments...
whether or not that slows things down as much, I don't know
--]]
function Tensor:lowerElem(indexes, lowers, m)
	if not m then m = self.metricInverse end
	return self:layerLowerElem({table.unpack(indexes)}, {table.unpack(lowers)}, m)
end

--[[
this removes the i'th dimension, summing across it

if it removes the last dim then a number is returned (rather than a 0-rank tensor, which I don't support)
--]]
function Tensor:contraction(i)
	assert(i >= 1 and i <= #self.dim, "tried to contract dimension "..i.." when we are only rank "..#self.dim)

	-- if there's a valid contraction and we're rank-1 then we're summing across everything
	if #self.dim == 1 then
		local result = 0
		for i=1,self.dim[1] do
			result = result + self:elem(i)
		end
		return result
	end

	local newdim = {table.unpack(self.dim)}
	local removedDim = table.remove(newdim,i)
	
	return Tensor(newdim, function(...)
		local indexes = {...}
		table.insert(indexes, i, 1)
		local result = 0
		for index=1,removedDim do
			indexes[i] = index
			result = result + self:elem(indexes)
		end
		return result
	end)
end


--[[
static method:

prepares any string-based symbolic representation of indexes
basically splits them into a table if they're a string
spaces are separators for using bigger-than-space index labels

T'ij' means upper-rank ij of two-rank tensor T
T'_ij' means lower-rank ij of two-rank tensor T
T{'_i','j'} means mixed-rank T_i^j

if we dereference with a table then by default all ranks are upper.
 In the case that we get a lower-rank prefix, only apply it to that individual element
 
If we dereference with a string then by default the ranks are upper,
Passing a '_' character will make all subsequent indexes lower, while
passing a '^' character will make all subsequent indexes upper.

T'i_j^k' means i and k on the top, j on the bottom
T'ij_kl^mn' means ij and mn on the top, kl on the bottom

This function returns an array of strings that are upper
unless they have a '_' prefix denoting that they are lower
--]]
function Tensor.prepareRepIndexes(indexes)

	local function handleTable(indexes)
		indexes = {table.unpack(indexes)}
		local comma = false
		for i=1,#indexes do
			if type(indexes[i]) == 'number' then
				indexes[i] = {
					number = indexes[i],
					comma = comma,
				}
			elseif type(indexes[i]) == 'table' and getmetatable(indexes[i]) == TensorIndex then
				indexes[i] = indexes[i]:clone()
			elseif type(indexes[i]) ~= 'string' then
				print("got an index that was not a number or string: "..type(indexes[i]))
			else
				local function removeIfFound(sym)
					local found = false
					while true do
						local symIndex = indexes[i]:find(sym,1,true)
						if symIndex then
							indexes[i] = indexes[i]:sub(1,symIndex-1) .. indexes[i]:sub(symIndex+#sym)
							found = true
						else
							break
						end
					end
					return found
				end
				-- if the expression is upper/lower..comma then switch order so comma is first
				if removeIfFound(',') then comma = true end
				local lower = not not removeIfFound('_')
				if removeIfFound('^') then
					--print('removing upper denotation from index table (it is default for tables of indices)')
				end
				-- if it has a '_' prefix then just leave it.  that'll be my denotation passed into TensorRepresentation
				if #indexes[i] == 0 then
					print('got an index without a symbol')
				end
				
				if tonumber(indexes[i]) ~= nil then
					indexes[i] = TensorIndex{
						number = tonumber(indexes[i]),
						lower = lower,
						comma = comma,
					}
				else
					indexes[i] = TensorIndex{
						symbol = indexes[i],
						lower = lower,
						comma = comma,
					}
				end
			end
		end
		return indexes	
	end

	if type(indexes) == 'string' then
		local indexString = indexes
		if indexString:find(' ') then
			indexes = handleTable(indexString:split(' '))
		else
			local lower = false
			local comma = false
			indexes = {}
			for i=1,#indexString do
				local ch = indexString:sub(i,i)
				if ch == '^' then
					lower = false 
				elseif ch == '_' then
					lower = true
				elseif ch == ',' then
					comma = true
				else
					if tonumber(ch) ~= nil then
						table.insert(indexes, TensorIndex{
							number = tonumber(ch),
							lower = lower,
							comma = comma,
						})
					else
						table.insert(indexes, TensorIndex{
							symbol = ch,
							lower = lower,
							comma = comma,
						})
					end
				end
			end
		end
	elseif type(indexes) == 'table' then
		indexes = handleTable(indexes)
	else
		error('indexes had unknown type: '..type(indexes))
	end
	
	for i,index in ipairs(indexes) do
		assert(index.number or index.symbol)
	end
	
	return indexes
end

--[[
returns a symbolic representation of a denotation of a tensor
if indexes is a string then this assumes each letter is a unique element in the representation
if indexes is a table then this assumes each number is a unique element in the representation

a:rep('ij') or a:rep{'i', 'j'} will both return a representation to tensor a with 'i' denoting the first dimension and 'j' denoting the second
--]]
function Tensor:rep(indexes)
	indexes = Tensor.prepareRepIndexes(indexes)

	for i,index in ipairs(indexes) do
		assert(index.number or index.symbol, "failed to find index on "..i.." of "..#indexes)
	end	
	
	local TensorRepresentation = require 'tensor.representation' 
	return TensorRepresentation(self, indexes)
end

-- shorthand lua: a'ij' works like a:rep('ij')
Tensor.__call = Tensor.rep

function Tensor:setMetric(t, dontGenerateInverse)
	self.metric = t
	assert(self.metric.rank == 2, "you can only set the metric to a rank-2 tensor")
	return self
end

function Tensor:setMetricInverse(t)
	self.metricInverse = t
	assert(self.metric.rank == 2, "you can only set the metric to a rank-2 tensor")
	return self
end

-- coordinate basis are currently only used for differentiation for comma derivatives
-- so pass a variable for comma derivative to apply diff()
-- and pass a function for it to apply that function
function Tensor:setCoordinateBasis(...)
	self.coordinateBasis = {...}
end

-- set the default that all tensors will use
do
	local eta = Tensor(4,4,function(i,j)
		if i == j then
			-- traditionally the 0th, sometimes the 4th, rarely the 1st of 4...
			if i == 1 then return -1 end
			return 1
		end
	end)
	Tensor:setMetric(eta)
	Tensor:setMetricInverse(eta)
end

-- function kronecherDelta -- we have upper and lower indexes to worry about ... 

function Tensor.ident(rank, dim)
	local dims = {}
	for i=1,rank do
		dims[i] = dim
	end
	return Tensor(dims, function(...)
		local is = {...}
		for i=1,#is-1 do
			if is[i] ~= is[i+1] then return 0 end
		end
		return 1
	end)
end

-- dim of each rank, and what that rank is
-- looks like this is spitting out the lower-rank Levi Civita when I want the upper rank version instead
-- solution? negative it
function Tensor.leviCivita(rank)
	local dims = {}
	for i=1,rank do
		dims[i] = rank
	end
	return Tensor(dims, function(...)
		local is = {...}

		-- now if any i's are matching then it's zero
		for i=1,#is-1 do
			for j=i+1,#is do
				if is[i] == is[j] then return 0 end
			end
		end
		
		-- before sorting, make sure 

		-- otherwise it's +1 for even permutations of the natural indexing, -1 for odd
		local swaps = 0
		for i=1,#is-1 do
			for j=#is-1,i,-1 do
				if is[j] > is[j+1] then
					is[j], is[j+1] = is[j+1], is[j]
					swaps = swaps + 1
				end
			end
		end
		
		-- negative the lower-index definition since our indexes are all upper by default
		if swaps % 2 == 1 then return 1 end
		return -1
	end)
end

return Tensor
