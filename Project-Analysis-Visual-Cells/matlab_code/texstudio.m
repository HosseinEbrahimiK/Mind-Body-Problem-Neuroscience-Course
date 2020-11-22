contents = dir();
for i=4 : 373
    if contents(i).name == "fget_spk.m" | contents(i).name == "fget_hdr.m" | contents(i).name == "texstudio.m~" | contents(i).name == "texstudio.m"
        break;
    end
    name = strcat(contents(i).name, ".csv");
    disp(contents(i).name);
    a = fget_spk(contents(i).name);
    csvwrite(name,a);
end