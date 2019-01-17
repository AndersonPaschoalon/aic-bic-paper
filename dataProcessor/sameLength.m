function [outvet1 outvet2] = sameLength(invet1, invet2)
% sameLength - make two vectors have the same length equals to the small
%
% Syntax:  [vet1 vet2] = sameLength(vet1, vet2)
%
% Inputs:
%    invet1 - first vector
%    invet2 - second vector
%
% Outputs:
%    outvet1 - first vector
%    outvet2 - second vector
%
% Example: 
%    [outvet1 outvet2] = sameLength([1 2 3 4], [1 2 3 4 5 6 7 8 9])
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% Author: Anderson Paschoalon
% email: anderson.paschoalon@gmail.com
% Sep 2018: Last revision: 16-Sep-2018 
	min_len = min(length(invet1), length(invet2));
	outvet1 = invet1(1:min_len);
	outvet2 = invet2(1:min_len);
end