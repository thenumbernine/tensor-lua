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
				local errmsg
				local result = {xpcall(func, function(err)
					errmsg = err .. '\n' .. debug.traceback()
				end)}	-- scope? all global, right? unless 'local' is added on...
				local duration = os.time() - startTime
				if errmsg then
					io.write(errmsg)
				else
					if #result > 0 then
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
