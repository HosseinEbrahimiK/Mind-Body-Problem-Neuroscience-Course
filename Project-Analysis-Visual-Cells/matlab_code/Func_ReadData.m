function a=Func_ReadData(directory)
pth= ['Address' directory];
dr=dir(pth);
list = [];
for i=3:length(dr)
    
    target_file1 = length(strfind(upper(dr(i).name),'msq1D.sa0'));
    target_file2 = length(strfind(upper(dr(i).name),'msq1d.sa0'));
    garbage_file1 = length(strfind(upper(dr(i).name),'.sa0.sub'));
    garbage_file2 = length(strfind(upper(dr(i).name),'.sa0.vecs'));
    
    if (target_file1 > 0 || target_file2 > 0) && garbage_file1 == 0 && garbage_file2 == 0
        
      [events, header]= fget_spk([pth '\' dr(i).name],'header');
      st.events= events;
      st.hdr= header;
      list =[list; st];
    end
end
