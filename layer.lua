local class = require 'ext.class'
local table = require 'ext.table'

local TensorLayer = class()

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
		indexes = {table.unpack(indexes[1])}
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
		indexes = {table.unpack(indexes[1])}
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
	indexes = {table.unpack(indexes)}
	lowers = {table.unpack(lowers)}
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

function TensorLayer.__concat(a,b) return tostring(a) .. tostring(b) end

return TensorLayer
