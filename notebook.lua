require 'tensor.tensor'

function asserteq(a,b)
	if a ~= b then
		error("\nexpected "..tostring(a).." to equal "..tostring(b))
	end
end

function notebook(cmd)
	for _,line in ipairs(cmd:split('\n')) do
		line = line:trim()
		if #line > 0 then
			print('> '..line)
			if line:sub(1,1) == '=' then
				line = 'return '..line:sub(2)
			end
			local startTime = os.time()	-- TODO hires timer
			local ok, err = assert(loadstring(line))
			if not ok then
				print(err)
			else
				local func = ok
				local result = {pcall(func)}	-- scope? all global, right? unless 'local' is added on...
				local duration = os.time() - startTime
				if #result == 2 and not result[1] then
					io.write(result[2]..'\n'..debug.traceback())
				else
					if #result > 0 or result[1] ~= nil then
						io.write(table.concat(table.map(result, tostring),'\t')..'\t')
					end
				end
				print('('..duration..' seconds)')
			end
		else
			print()
		end
	end
end
