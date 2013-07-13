require 'tensor.tensor'
require 'tensor.notebook'

Tensor, leviCivita = tensor.Tensor, tensor.leviCivita

notebook[[

ident4 = Tensor(4,4,function(i,j) return ({[0]=1})[i-j] end)

eps4 = leviCivita(4)
eps4:setMetric(ident4)	-- delta4_1 evaluates to the metric tensor of eps4, be this ident4 or the minkowski metric...

delta4_3 = (-eps4'abcd' * eps4'_uvwd'):setMetric(ident4)'abc_uvw'

delta4_2 = (1/2 * delta4_3'abc_uvc'):setMetric(ident4)'ab_uv'
delta4_2b = (-1/2 * eps4'abcd' * eps4'_uvcd'):setMetric(ident4)'ab_uv'

=delta4_2 == delta4_2b

delta4_1 = (1/3 * delta4_2'ab_ub'):setMetric(ident4)'a_u'

delta4_1b = (1/6 * delta4_3'abc_ubc'):setMetric(ident4)'a_u'
delta4_1c = (-1/6 * eps4'abcd' * eps4'_ubcd'):setMetric(ident4)'a_u'

=delta4_1 == delta4_1b
=delta4_1 == delta4_1c

]]

--[[
here's an important question...

if a and b are rank-2 tensors
and c^i_k = a^i_j * b^j_k
and c'^i^k = a^i^j * b^j^k (relaxing the constraint that you can only sum across adjacent indices)
then how are the components of c related to c'?
c^i_k = a^i_j * b^j_k
c^i_k * eta^k^a = a^i_j * b^j_k * eta^k^m
c^i^m = a^i_j * b^j^m
c^i^m = (a^i^n * eta_n_j) * b^j^m
... so in matrix representation, we'd be rhs multiplying a by the metric tensor before multiplying a & b ...
... and therefore the coefficients would indeed by different
the issue only arises with mixed notation.  if all indexes are lower then the etas cancel out and matrix mul remains the same

this is where my issues with the kronecker delta come into play:
if you raise or lower a delta index it is impervious to the metric tensor
this makes delta identities described in mixed index notation a bit frustrating to input ...

unless delta's metric, specifically, is changed to be the identity matrix rather than the minkowski metric ...
then things work out, except upon multiplying a delta with another tensor, how do we specify what metric the result gets? A or B?


in other news ... is g^i^j really the inverse of g_i_j if we use g to transform both its indices?
it can already be shown that, if by definition of delta, g^i^j * g_j_k = delta^i_k, and if by definition of index raising,
 g^i_k = g^i^j * g_j_k, then combining the two definitions gives g^i_k = delta^i_k.  same can be reached for g_i^k = delta_i^k

so applying raising to both of g's ... a^u^v := (by raising index definition) 
g^u^i * a_i_j * g^j^v = (highG) * A * (highG) for (highG)ij = g^i^j and for (a_i_j)ij = (A)ij
...so if we replace a with g, we get g^u^v := g^u^i * g_i_j * g^j^v = highG * G * highG
In matrix form we're left with highG = highG * G * highG, whic is equivalent to Ident = G * highG and Ident = highG * G
which is true for highG = G^-1.  viola.


here's another: does (a^i_j)ij = (a_j^i)ij?
do the components of the two tensors match?  does the left-right order matter, or just the top-bottom order?
a^i_j = a^i^k * g_k_j = A * G, so (a^i_j)ij = (A * G)ij
a_j^i = g_j_k * a^k^i = G * A, so (a_j^i)ij = ((G * A)^T) = A^T * G^T
so the two are equal only when A and G are symmetric.  This is always true for G.  not always true for A ...

a^i_j = g^i^u * a_u_j = a^i^u * g_u_j :: Ahilo = G^-1 * Alolo = Ahihi * G
a_j^i = g_j_u * a^u^i = a_j_u * g^u^i :: Alohi = G * Ahihi = Alolo * G^-1
... so ...
Alohi = Alolo * G^-1
Alohi = G * Ahihi
Ahilo = G^-1 * Alolo
Ahilo = Ahihi * G
... so let Ahihi = A (the "real" A) ... we want to see if Ahilo = Alohi^T
A = Alohi * G = G * Ahilo
Alohi = G * Ahilo * G^-1
Alohi^T = (G * Ahilo * G^-1)^T = G^-1^T * Ahilo^T * G^T
Alohi^T = G^-1 * Ahilo^T * G^T

it looks like the safest definition of the kronecher delta is delta^i_j = dx^i/dx^j ... which is eta ... so delta is in fact eta?
--]]