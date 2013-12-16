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

require 'ext'

module('tensor', package.seeall)

local function concatToString(a,b)
	return tostring(a) .. tostring(b)
end


TensorLayer = class()

function TensorLayer:__tostring()
	local s = {}
	for i=1,self.layerDim do
		table.insert(s, tostring(self[i]))
	end
	return '{' .. table.concat(s, ',') .. '}'
end

--[[
indexes is an array size of the tensor rank
the array contains numbers
returns the element at the specified location in the tensor

a:elem(1,2,3) or a:elem{1,2,3} work
--]]
function TensorLayer:elem(...)
	local indexes = {...}
	if type(indexes[1]) == 'table' then 
		indexes = {unpack(indexes[1])}
	end
	local index = table.remove(indexes, 1)
	local value = self[index]
	if #indexes > 0 then
		return value:elem(indexes)
	else
		return value
	end
end

--[[
t:setElem(i1,..,iN, value)
t:setElem({i1,..,iN}, value)
--]]
function TensorLayer:setElem(...)
	local indexes = {...}
	local value
	if type(indexes[1]) == 'table' then
		value = indexes[2]
		indexes = {unpack(indexes[1])}
	else
		value = table.remove(indexes)
	end
	
	local index = table.remove(indexes, 1)
	if #indexes > 0 then
		self[index]:setElem(indexes, value)
	else
		self[index] = value
	end
end

function TensorLayer:layerLowerElem(indexes, lowers, m)
	indexes = {unpack(indexes)}
	lowers = {unpack(lowers)}
	local index = table.remove(indexes, 1)
	local lower = table.remove(lowers, 1)
	if lower then
		local value = 0
		for i=1,#self do
			local selfValue
			if #indexes > 0 then
				selfValue = self[i]:layerLowerElem(indexes, lowers, m)
			else
				selfValue = self[i]
				assert(type(selfValue) == 'number')
			end
			value = value + m:elem(index, i) * selfValue
		end
		return value
	else
		local value = self[index]
		if #indexes > 0 then
			return value:layerLowerElem(indexes, lowers, m)
		else
			assert(type(value) == 'number')
			return value
		end
	end
end


TensorLayer.__concat = concatToString


Tensor = class(TensorLayer)

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
					v = values(unpack(indexes))
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
	print(unpack(indexes))
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
		return symmath.diff(self:elem(...), unpack(diffVars))
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
	
	local newdim = {unpack(self.dim)}
	-- remove the second index from the new dimension
	local removedDim = table.remove(newdim,j)
	-- keep track of where the first index is in the new dimension
	local newdimI = i
	if j < i then newdimI = newdimI - 1 end
	
	return Tensor(newdim, function(...)
		local indexes = {...}
		-- now when we reference the unremoved dimension

		local srcIndexes = {unpack(indexes)}
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
			local vis = {unpack(is)}
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
	return self:layerLowerElem({unpack(indexes)}, {unpack(lowers)}, m)
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

	local newdim = {unpack(self.dim)}
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

local TensorIndex = class()

function TensorIndex:init(args)
	self.lower = args.lower or false
	self.comma = args.comma or false
	self.symbol = args.symbol
	assert(type(self.symbol) == 'string' or type(self.symbol) == 'nil')
	self.number = args.number
	assert(type(self.number) == 'number' or type(self.number) == 'nil')
end

function TensorIndex.clone(...)
	return TensorIndex(...)	-- convert our type(x) from 'table' to 'function'
end

function TensorIndex.__eq(a,b)
	return a.lower == b.lower
	and a.comma == b.comma
	and a.symbol == b.symbol
	and a.number == b.number
end

function TensorIndex:__tostring()
	local s = ''
	if self.comma then s = ',' .. s end
	if self.lower then s = '_' .. s else s = '^' .. s end
	if self.symbol then
		return s .. self.symbol
	elseif self.number then
		return s .. self.number
	else
		error("TensorIndex expected a symbol or a number")
	end
end

function TensorIndex.__concat(a,b)
	return tostring(a) .. tostring(b)
end


--[[
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
local function prepareRepIndexes(indexes)

	local function handleTable(indexes)
		indexes = {unpack(indexes)}
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
	indexes = prepareRepIndexes(indexes)

	for i,index in ipairs(indexes) do
		assert(index.number or index.symbol, "failed to find index on "..i.." of "..#indexes)
	end	
	
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


TensorRepresentation = class()

--[[
construct a representation of a tensor with indexes denoting dimensions to operate on
these are typically built with the help of the Tensor:rep()
 TODO - rather than tensor product + trace, combine the two into one (to save on memory)
--]]
function TensorRepresentation:init(tensor, indexes)
	assert(tensor, "TensorRepresentation expected a tensor")
	assert(indexes, "TensorRepresentation expected indexes")

	for i,index in ipairs(indexes) do
		assert(index.number or index.symbol, "failed to find index on "..i.." of "..#indexes)
	end	

--print('rep provided with tensor '..tensor..' and indexes '..table(indexes):map(tostring):concat())
	
	local commaIndexes = {}
	local nonCommaIndexes = {}
	for i,index in ipairs(indexes) do
		if index.comma then
			table.insert(commaIndexes, i)
		else
			table.insert(nonCommaIndexes, i)
		end
	end

	
	-- make sure indexes match rank
	if not isTensor(tensor) then
		if #indexes > 0 then
			error("tried to apply "..#indexes.." indexes to a 0-rank tensor (a scalar): "..tostring(tensor))
		end
		if #nonCommaIndexes ~= 0 then
			error("Tensor.rep non-tensor needs as zero non-comma indexes as the tensor's rank.  Found "..#nonCommaIndexes.." but needed "..0)
		end
	else
		if #nonCommaIndexes ~= tensor.rank then
			error("Tensor.rep needs as many non-comma indexes as the tensor's rank.  Found "..#nonCommaIndexes.." but needed "..tensor.rank)
		end
	end
	
	-- this operates on indexes
	-- which hasn't been expanded according to commas just yet
	-- so commas must be all at the end
	local function transformIndexesToUpper(withComma)
		-- raise all indexes, transform tensors accordingly
		for i=1,#indexes do
			if indexes[i].lower
			and indexes[i].comma == withComma
			then
				if tensor.dim[i] ~= tensor.metricInverse.dim[1] then
					print("can't lower index "..i.." until you set the metric tensor to one with dimension matching the tensor you are attempting to raise/lower")
					print("  your tensor's dimensions: "..table.concat(tensor.dim,','))
					print("  metric tensor's dimensions: "..table.concat(tensor.metricInverse.dim,','))
					error("you can reset the metric tensor via the Tensor.setMetric() function")
				end
			
				-- TODO transform after finding numbers (or will this numbers found affect transformations?)
				-- that will mean if we get a lower number dereference then we'll have to transform that index number while removing it...
				tensor = tensor:transformIndex(i, tensor.metricInverse)
				indexes[i].lower = false
--print('after transform: '..tensor..' and indexes '..table(indexes):map(tostring):concat())
			end
		end
	end
	
	transformIndexesToUpper(false)

	-- if any are commas then ... expand the gradients and clear the commas
	-- do this before testing rank so that we can gather our new rank
	--[[
	comma derivative notes:
	(g_ik * a^k),_j == g_ik * (a^k,_j) ... is true only for g orthonormal and constant.  otherwise the , distributes through and we get
	(g_ik * a^k),_j = g_ik,_j * a^k + g_ik * a^k,_j ... and see how that g_ik,j fits in there as the skew to the comma distribution?
	(g_ik * a^k);_j = g_ik;_j * a^k + g_ik * a^k;_j = g_ik * a^k;_j ... since g_ik;j = 0 for the metric (courtesy of index raising being based on the metric inverse)
	
	so now that we know a lowered derivative is different than a derivative lowered ...
	... which do we want to accept as the standard?
	--]]
	if #commaIndexes > 0 then
		local newdim = {unpack(tensor.dim)}
		for i=1,#commaIndexes do
			table.insert(newdim, #tensor.coordinateBasis)
		end
--print('before differentiation: '..tensor)
		assert(tensor.coordinateBasis, "cannot use comma derivative without a coordinate basis")
		-- non-coordinate (anholonomic) basis use a bit more than just the derivative
		assert(#tensor.coordinateBasis == tensor.metric.dim[1], "need the same number of coordinate basis as is our metric tensor's dimensions")
		tensor = Tensor(newdim, function(...)
			local is = {...}
			-- pick out 
			local base = {}
			local deriv = {}
			for i=1,#is do
				if indexes[i].comma then
					table.insert(deriv,is[i])
				else
					table.insert(base,is[i])
				end
			end
			local x = tensor:elem(unpack(base))
			for i=1,#deriv do
				local basis = tensor.coordinateBasis[deriv[i]]
				assert(type(basis) == 'function', "for now I only allow operations as coordinate basis")
				x = basis(x)
			end
			return x
		end)

		transformIndexesToUpper(true)
		
		for i=1,#indexes do
			indexes[i].comma = false
		end
--print('after differentiation: '..tensor)
	end
	

	-- pick out specific number indexes
	-- much like assign()'s permutation operation
	do
		local foundNumbers
		for i=1,#indexes do
			if indexes[i].number then
				foundNumbers = true
			end
		end
		if foundNumbers then
			local newdim = {unpack(tensor.dim)}
			local srcIndexes = {unpack(indexes)}
			local sis = {}
			for i=#tensor.dim,1,-1 do
				if indexes[i].number then
					sis[i] = indexes[i].number	-- remember it for accessing source elements later
					table.remove(indexes, i)
					table.remove(newdim, i)
				end
			end
			local srcTensor = tensor
			if #newdim == 0 then
				-- then all of sis should be filled, so just use the number
				tensor = srcTensor:elem(sis)
			else
				local dstToSrc = {}
				for i=1,#newdim do
					dstToSrc[i] = assert(table.find(srcIndexes, indexes[i]))
				end
				tensor = Tensor(newdim, function(...)
					local is = {...}
					for i=1,#is do
						sis[dstToSrc[i]] = is[i]
					end
					return srcTensor:elem(sis)
				end)
				
--print('after contraction: '..tensor..' and indexes '..table(indexes):map(tostring):concat())
				
			end
		end
	end	

	self.tensor = assert(tensor)
	self.indexes = setmetatable(table.map(indexes, TensorIndex.clone), nil)
	
	-- for all indexes
	
	-- apply any summations upon construction
	-- if any two indexes match then zero non-diagonal entries in the resulting tensor
	--  (scaling with the delta tensor)
	
	local modified
	repeat
		modified = false
		for i=1,#self.indexes-1 do
			for j=i+1,#self.indexes do
				if self.indexes[i] == self.indexes[j] then
					self.tensor = self.tensor:trace(i,j)
					table.remove(self.indexes,j)	-- remove one of the two matching indices
					modified = true
					break
				end
			end
			if modified then break end
		end
	until not modified
	
	
	for i,index in ipairs(self.indexes) do
		assert(index.number or index.symbol, "failed to find index on "..i.." of "..#self.indexes)
	end	
end

--[[
a'ik' * b'kj'
TensorRepresentation(a, {'i', 'k'}) * TensorRepresentation(b, {'k', 'j'})

a_ik * b_kj = a_ik * delta_kl * b_lj
a_ik * b_kj = a_ik * b_lj * delta_kl
c_il = a_ijk * b_jkl = a_ijk * b_mno * delta_jm * delta_kn
c_ij = a_ijikil = a_ijklmn * delta_ik * delta_im ... and then sum across l and n

a_ij * b_kl = an outer product of a and b
a_ij * b_jl = (a_ij * b_kl) * delta_jk = outer product then contraction of j & k
	...produces a rank-3 tensor (a*b)_ijl
	...so to get from (a*b)_ijl to c_il = a_ij * b_jl, we sum across 'l'

will produce a new tensor of all combined ranks (a(1), a(2), b(1), b(2) in this case)
then scale with delta across any similar ranks (a(2) and b(1) in this case)
(mind you, allocation then summation means for a 4x4 * 4x4 we're making 256 entries before summing back down to 64)

TensorRepresentation(c, {'i', 'k', 'j'})	... or with indexes in some permutation

c = (a'ik' * b'kj')'ij'  is our equivalent of c_ij = a_ik * b_kj

the final TensorRepresentation.assignTo(indexes) will assign the tensor and sum any indexes left out

Traces are a:rep('ii') which will return a rank-0 tensor
--]]
function TensorRepresentation.__mul(a,b)
	local ta = getmetatable(a) == TensorRepresentation
	local tb = getmetatable(b) == TensorRepresentation
	if ta and tb then
		-- rather than outer product and reduce in the TensorRepresentation constructor,
		-- do the reduction up front
		-- this will save us orders of magnitude of allocated memory
		--return TensorRepresentation(a.tensor * b.tensor, table():append(a.indexes):append(b.indexes))
		-- this is probably going to look very similar to the constructor ...
		
		--[[
		pick out all number, lower, or paired indexes
		returns a new table of tables with the following:
			index (as a number if it is a number)
			lower, set to 'true' if the index was lowered
			matches, set to 'true' if this matches other indexes in either A or B
		--]]
		local function getIndexStats(rep)
			local indexes = rep.indexes
			local tensor = rep.tensor
			local stats = {}
			-- 4x4 inverse, transforming up front: 79 seconds, transforming later: 87 seconds
			--[[ transform later
			local lowers = {}
			--]]
			for i,index in ipairs(indexes) do
				--[[ transform later
				lowers[i] = false
				--]]
				stats[i] = {}
				if index.lower then
					if tensor.dim[i] ~= tensor.metricInverse.dim[1] then
						print("can't lower index "..i.." until you set the metric tensor to one with dimension matching the tensor you are attempting to raise/lower")
						print("  your tensor's dimensions: "..table.concat(tensor.dim,','))
						print("  metric tensor's dimensions: "..table.concat(tensor.metricInverse.dim,','))
						error("you can reset the metric tensor via the Tensor.setMetric() function")
					end
				
					-- [[ transform up front
					tensor = tensor:transformIndex(i, tensor.metricInverse)
					--]]
					--[[ transform later
					lowers[i] = true
					--]]
				end
				stats[i].index = index.symbol or index.number
			end
			-- [[ transform up front
			return stats, tensor
			--]]
			--[[ transform later
			return stats, tensor, lowers
			--]]
		end
		-- [[ transform up front
		local statsA, tensorA = getIndexStats(a)
		local statsB, tensorB = getIndexStats(b)
		--]]
		--[[ transform later
		local statsA, tensorA, lowersA = getIndexStats(a)
		local statsB, tensorB, lowersB = getIndexStats(b)
		--]]
		
		-- now calculate pairs
		local indexPairs = {}
		for ai,sa in ipairs(statsA) do
			assert(sa.index)
			for bi,sb in ipairs(statsB) do
				if type(sa.index) ~= 'number' and sa.index == sb.index then
					assert(tensorA.dim[ai] == tensorB.dim[bi], "you can't pair indexes of unmatching dimension!")
					if not indexPairs[sa.index] then 
						indexPairs[sa.index] = {
							alocs={},
							blocs={},
							dim=tensorA.dim[ai],
						}
					end
					table.insertUnique(indexPairs[sa.index].alocs,ai)
					table.insertUnique(indexPairs[sb.index].blocs,bi)
				end
			end
		end
		
		local newIndexes = {}
		local newdim = {}
		local foundPairs = {}
		local function processStats(stats, dim, srcIndexes)
			for i,st in ipairs(stats) do
				-- prep our source indexes with whatever fixed numbers we have
				-- also cuts down on reallocation to only alloc them once
				if type(st.index) == 'number' then
					srcIndexes[i] = st.index
				else
					local p = indexPairs[st.index]
					local skip = false
					if p then
						-- add the first of a set of matching pairs, but none after the first
						if not foundPairs[st.index] then
							foundPairs[st.index] = true
						else
							skip = true
						end
					end
					if not skip then
						local indexNumber, indexSymbol
						if type(st.index) == 'string' then indexSymbol = st.index end
						if type(st.index) == 'number' then indexNumber = st.index end
						table.insert(newIndexes,TensorIndex{
							symbol = indexSymbol,
							number = indexNumber,
							lower = st.lower,
						})
						if foundPairs[st.index] then	-- ...then we're adding the first of a set of pairs
							foundPairs[st.index] = #newIndexes
						end
						st.dstIndex = #newIndexes
						table.insert(newdim,dim[i])
					else
						st.dstIndex = foundPairs[st.index]
					end
				end
			end
		end
		local ais = {}
		processStats(statsA, tensorA.dim, ais)
		local bis = {}
		processStats(statsB, tensorB.dim, bis)
		
		-- now we have lower information, number information, match information
		-- now we can create the tensor
		return TensorRepresentation(Tensor(newdim, function(...)
			local is = {...}
			
			--[[
			if any source indexes are numbers then fix those coordinates
			if any source indexes are matching then calculate the sum of products of all matching indexes
			--]]
			for i,sa in ipairs(statsA) do
				if type(sa.index) ~= 'number' then
					ais[i] = is[sa.dstIndex]
				end
			end
			for i,sb in ipairs(statsB) do
				if type(sb.index) ~= 'number' then
					bis[i] = is[sb.dstIndex]
				end
			end

			
			-- [[ transform up front
			local avalue = tensorA:elem(ais)
			local bvalue = tensorB:elem(bis)
			--]]
			--[[ transform later
			local avalue = tensorA:lowerElem(ais, lowersA)
			local bvalue = tensorB:lowerElem(bis, lowerB)
			--]]
			return avalue * bvalue
		end), newIndexes)
		
	elseif ta then
		return TensorRepresentation(a.tensor * b, a.indexes)
	elseif tb then
		return TensorRepresentation(a * b.tensor, b.indexes)
	end
end

local function commonIndices(indexesA, indexesB)
	local indexes = {unpack(indexesA)}	-- start with A, filter out what's not in B
	for i=#indexes,1,-1 do
		if not table.find(indexesB, indexes[i]) then table.remove(indexes, i) end
	end
	return indexes
end

--[[
addition ... what if the indexes don't match up?
a'ij' + b'ji'
well this would currently be represented as (a'ij''ij' + b'ij''ji')
but ... how can we get that to be produced by (a'ij' + b'ji')'ij' ?
 well, take the first index, permute the second, store the representation as the first's index

how about a_ij + b_ii ?
technically that only works if each term is summed over all before being added
 = sum(i,j) a_ij + sum(i) b_ii

first permute the second tensor representation's indexes to match the first
(and the tensor itself) this should only permute
no such summations should occur that didn't already from ctor'ing the original b
next do the addition

actually, if one term has indexes that another doesn't, we should be appending those dimensions
so that a_i + b_j is rank 2 ... or should we? that doesn't seem right.
at the moment it says "a uses 'i'" and then looks a 'b', and contracts its non-'i' indices
...but it should be contracting all indices found in neither (not just the ones in 'b' not in 'a')
--]]
function TensorRepresentation.__add(a,b)
	local indexes = commonIndices(a.indexes, b.indexes)
	return TensorRepresentation(a:assign(indexes) + b:assign(indexes), indexes)
end

function TensorRepresentation.__sub(a,b)
	local indexes = commonIndices(a.indexes, b.indexes)
	return TensorRepresentation(a:assign(indexes) - b:assign(indexes), indexes)
end

function TensorRepresentation.__div(a,b)
	return TensorRepresentation(a.tensor / b, a.indexes)
end

function TensorRepresentation.__unm(a)
	return TensorRepresentation(-a.tensor, a.indexes)
end

-- implicit sum across all non-common indexes?
-- or should I make users explicitly write representations before testing equality?
function TensorRepresentation.__eq(a,b)
	local indexes = commonIndices(a.indexes, b.indexes)
	a = a:assign(indexes)
	b = b:assign(indexes)
	return a == b
end

function TensorRepresentation:__tostring()
	return '['..tostring(self.tensor)..']'..table.map(self.indexes, tostring):concat()
end

TensorRepresentation.__concat = concatToString

function isTensor(t)
	if type(t) ~= 'table' then return false end
	if not t.isa then return false end
	return t:isa(Tensor)
end

--[[
if any indexes in here aren't mentioned then sum across them 
none should be repeated (the stmt "c_ii = a_i" doesn't fully define c, does it?)
--]]
function TensorRepresentation:assign(indexes)
	if not indexes then
		indexes = {}
	else
		indexes = prepareRepIndexes(indexes)
	end
	
	for _,index in ipairs(indexes) do
		assert(not index.comma, "can't assign to comma indexes")
	end

	if not isTensor(self.tensor) then
		if #indexes ~= 0 then
			error("tried to index a 0-rank tensor (a scalar): "..tostring(self.tensor).." with indexes "..table.map(indexes,tostring):concat())
		end
		return self.tensor
	end
	
	local indexesToTransform = {}
	for i=1,#indexes do
		-- if we're assigning an upper index to a lower index
		-- then we want to transform the upper by the metric tensor
		-- (not its inverse, as I transform to before operations are carried out)
		if indexes[i].lower then
			indexes[i].lower = false
			table.insert(indexesToTransform, i)
		end
	end
	
	local srcTensor = self.tensor:clone()
	local srcIndexes = setmetatable(table.map(self.indexes, TensorIndex.clone), nil)
	for i=#srcIndexes,1,-1 do
		local index = srcIndexes[i]
		if not table.find(indexes, index) then
			srcTensor = srcTensor:contraction(i)
			table.remove(srcIndexes, i)
		end
	end
	
	-- no need to relabel indexes if we have nothing left
	if #srcIndexes == 0 then
		assert(not isTensor(srcTensor))		--assert(type(srcTensor) == 'number')
		return srcTensor
	end
	
	--[[
	now we need to permute from srcIndexes to indexes
	so get a mapping from indexes to srcIndexes
	--]]

	local dstToSrcIndexes = {}
	--[[
	for i,index in ipairs(indexes) do
		dstToSrcIndexes[i] = assert(table.find(srcIndexes, index), "make sure all your assigned indexes appear in the tensor expression")
	end
	--]]
	for i,index in ipairs(srcIndexes) do
		dstToSrcIndexes[i] = assert(table.find(indexes, index), "make sure all your assigned indexes appear in the tensor expression")
	end
	
	local result = Tensor(srcTensor.dim, function(...)
		local is = {...}
		-- now remap from the dest indexes to the srcTensor's indexes
		local sis = {}
		for i=1,#is do
			-- we have the i'th index in the destination element
			-- we need the j'th index in the source element
			sis[i] = is[dstToSrcIndexes[i]]
		end
		return srcTensor:elem(sis)
	end)
	
	-- TODO fix this ...
	--result:setMetric(srcTensor.metric)
	result.metric = srcTensor.metric
	result.metricInverse = srcTensor.metricInverse
	
	--[[
	b_i^j = a^ij
	
	--]]
	for _,ix in ipairs(indexesToTransform) do
		result = result:transformIndex(ix, result.metric)
	end
	
	return result
end

-- shorthand for reassignment: 
-- c = (a:rep('ik') * b:rep('kj')):assign('ij') becomes (a'ik' * b'kj')'ij'
TensorRepresentation.__call = TensorRepresentation.assign

function TensorRepresentation:setMetric(...) 
	self.tensor:setMetric(...)
	return self	-- C++ hackery bad habit syntax
end

--[[
matrix mul
[a11 a12] [b11 b12]   [a11*b11+a12*b21 a11*b12+a12*b22]
[a21 a22]*[b21 b22] = [a21*b11+a22*b21 a21*b12+a22*b22]

outer product
[a11 a12] [b11 b12]
[a21 a22]o[b21 b22] = 

 [a11*b11 a12*b11]|[a11*b12 a12*b12]
 [a21*b11 a22*b11]|[a21*b12 a22*b12]
 -----------------+-----------------
 [a11*b21 a12*b21]|[a11*b22 a12*b22]
 [a21*b21 a22*b21]|[a21*b22 a22*b22]

...times delta of a's 2nd and b's 1st...
(meaning zero the terms that a's 2nd and b's 1st don't match) 

 [a11*b11 0]|[a11*b12 0]
 [a21*b11 0]|[a21*b12 0]
 -----------+-----------
 [0 a12*b21]|[0 a12*b22]
 [0 a22*b21]|[0 a22*b22]
 
then sum the matching terms
 
 [a11*b11+a12*b21 a11*b12+a12*b22]
 [a21*b11+a22*b21 a21*b12+a22*b22]
 
...without applying the deltas we get ...

 [a11*b11+a12*b11+a11*b21+a12*b21 a11*b12+a12*b12+a11*b22+a12*b22]
 [a21*b11+a22*b11+a21*b21+a22*b21 a21*b12+a22*b12+a21*b22+a22*b22]
 
1 2   1 2    7 10
3 4 * 3 4 = 15 22

1 2 3   1 2 3    30  36  42
4 5 6 * 4 5 6 =  66  81  96
7 8 9   7 8 9   102 126 150

how about aij*bij?
[a11 a12] [b11 b12]
[a21 a22]o[b21 b22] = 

 [a11*b11 a12*b11]|[a11*b12 a12*b12]
 [a21*b11 a22*b11]|[a21*b12 a22*b12]
 -----------------+-----------------
 [a11*b21 a12*b21]|[a11*b22 a12*b22]
 [a21*b21 a22*b21]|[a21*b22 a22*b22]

[a11*b11 a12*b12]
[a21*b21 a22*b22]

how about aii*bjj?
(a11 + a22) * (b11 + b22)
a11*b11 + a11*b22 + a22*b11 + a22*b22
 
--]]

-- function kronecherDelta -- we have upper and lower indexes to worry about ... 

function ident(rank, dim)
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
function leviCivita(rank)
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
