## Copyright (C) 2017 anderson
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} cumulativeData (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: anderson <anderson@duvel-ThinkCentre-M93p>
## Created: 2017-04-12

function [retVet] = cumulativeData (dataVet)
% cumulativeData - create a cumulative data vector, acumulating dataVet 
%
% Syntax: retVet = cumulativeData (dataVet)
%
% Inputs:
%    dataVet - original vector
%
% Outputs:
%    dataVet - acumulated vector
%
% Example: 
%    retVet = cumulativeData ([1 2 3])
%    % retVet has [1 3 6]
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% Author: Anderson Paschoalon
% email: anderson.paschoalon@gmail.com
% Sep 2018: Last revision: 16-Sep-2018 
	m = length(dataVet);
	%init retVet
	retVet = zeros(m, 1);
	
	for i = 1:m
		if(i == 1)
			retVet(i) = dataVet(i);
		elseif
			retVet(i) = dataVet(i) + retVet(i - 1);
		endif
		
	endfor

endfunction
