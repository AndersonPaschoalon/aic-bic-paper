function sff2File(filename, title, labels, separator, vs, vf1, vf2)
	fid = fopen(filename,'wt');
	fprintf(fid, "#%s\n", title);
	fprintf(fid, "#%s\n", labels;
	m = length(v1)
	for i = 1:m
		fprintf(fid, "%s%c%f%c%f\n", vs(i), separator, vf1(i), separator, vf2(i));
	endfor
	%dlmwrite(fid, M, "-append");
	fclose(fid)
end