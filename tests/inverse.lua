--[[

    File: inverse.lua 

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
