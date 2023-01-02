images := castle snail

%: %.cu 
	nvcc support.cu $< -o $@
	$@ images/castle.pnm 500

clean:
	rm -f images/*host* images/*device* *.exe *.exp *.lib