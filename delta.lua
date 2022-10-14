local Tensor = require 'tensor'

--[[
TODO - for arbitrary subdim of the dim (i.e. 4_4, 4_3, 4_2, 4_1)
--]]
local function kroneckerDelta(dim, subdim)
	if not subdim then subdim = dim end
	local dims = {}
	for i=1,subdim*2 do
		dims[i] = dim
	end
	return Tensor(dims,function(...)
		local is = {...}
		local his = {}
		local los = {}
		for i=1,subdim do
			his[i] = is[i]
			los[i] = is[i+subdim]
		end
		local function checkMatches(list)
			for i=1,#list-1 do
				for j=i+1,#list do
					if list[i] == list[j] then
						return i,j
					end
				end
			end
		end
		if checkMatches(his) then return 0 end
		if checkMatches(los) then return 0 end

		local swap = 0
		for i=1,#his-1 do
			for j=#his-1,i,-1 do
				if his[j] > his[j+1] then
					his[j], his[j+1] = his[j+1], his[j]
					swap = swap + 1
				end
				if los[j] > los[j+1] then
					los[j], los[j+1] = los[j+1], los[j]
					swap = swap + 1
				end
			end
		end
		
		-- if, sorted, they aren't equal, then return 0
		for i=1,subdim do
			if his[i] ~= los[i] then return 0 end	
		end
		
		-- if our first index is 2 then add 1
		-- if our first index is 3 then add 1
		-- something seems off ...
		if his[1] ~= 1 then swap = swap + 1 end		-- erm... why is this working?
		
		-- return negative so half the lows being on the bottom will change the sign ('^12_12' gets 1, so '^12^12' gets -1)
		local result = -1
		if swap % 2 == 1 then result = 1 end
		--local correct = delta4_3
		--print(table.concat(is,',')..' sorted: '..table.concat(his,',')..', '..table.concat(los,',')..' swaps '..swap..' is '..result..' should be '..correct:elem(is)..(({[0] = ''})[result - correct:elem(is)] or ' is off!'))
		return result
	end)
end

return kroneckerDelta
