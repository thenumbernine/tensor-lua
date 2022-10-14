--[[
TODO all tensor should be representations
and should store their underlying representations (no more 'natural' corvariant storage)
--]]

local class = require 'ext.class'
local table = require 'ext.table'

-- this is interesting ... require 'tensor' via package.path = ?/?.lua which then requires this
--  which then requires 'tensor.tensor' comes out as a different package ...
-- so to ensure that require 'tensor' and require 'tensor.tensor' are the same ...
--  you should put another level of indirection between the resolved name and the content
-- (hence the original purpose of 'module')
local Tensor = require 'tensor'	-- this is assuming that whoever required 'tensor' did so this way and not require 'tensor.tensor'
local TensorIndex = require 'tensor.index'

local TensorRepresentation = class()

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
	if not Tensor:isa(tensor) then
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
		local newdim = {table.unpack(tensor.dim)}
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
			local x = tensor:elem(table.unpack(base))
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
			local newdim = {table.unpack(tensor.dim)}
			local srcIndexes = {table.unpack(indexes)}
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
	local indexes = {table.unpack(indexesA)}	-- start with A, filter out what's not in B
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

function TensorRepresentation.__concat(a,b)
	return tostring(a) .. tostring(b)
end

--[[
if any indexes in here aren't mentioned then sum across them 
none should be repeated (the stmt "c_ii = a_i" doesn't fully define c, does it?)
--]]
function TensorRepresentation:assign(indexes)
	if not indexes then
		indexes = {}
	else
		indexes = Tensor.prepareRepIndexes(indexes)
	end
	
	for _,index in ipairs(indexes) do
		assert(not index.comma, "can't assign to comma indexes")
	end

	if not Tensor:isa(self.tensor) then
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
		assert(not Tensor:isa(srcTensor))		--assert(type(srcTensor) == 'number')
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

return TensorRepresentation
