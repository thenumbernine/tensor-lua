require 'tensor.notebook'
require 'tensor.matrix'

Tensor, leviCivita = tensor.Tensor, tensor.leviCivita

notebook[[
eta = Tensor(4,4,function(i,j) if i ~= j then return 0 end if i == 1 then return -1 end return 1 end)
Tensor:setMetric(eta)
=eta
=det(eta)
=inv(eta)
a = Tensor(2,2,function(i,j) return j+(i-1)*2 end)
=det(a)
a = Tensor(3,3,function(i,j) return j+(i-1)*3 end)
=det(a)
a = Tensor(4,4,function(i,j) return j+(i-1)*4 end)
=a
=a"ii"
=a"i1"
=a"1i"
=det(a)
=inv(a)
t=Tensor(4,4,{{0.5,-0,-0,0.5},{-0,0.5,0.5,-0},{-0,-0.5,0.5,-0},{-0.5,-0,-0,0.5}})
=t
invT=inv(t)
=invT
correctInvT=Tensor(4,4,{{1,0,0,-1},{0,1,-1,0},{0,1,1,0},{1,0,0,1}})
=correctInvT
=invT==correctInvT
]]
