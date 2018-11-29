clear all; close all; clc;
load digitStruct.mat
Images = zeros(64,64,3,length(digitStruct),'uint8');
Labels = cell(length(digitStruct), 1);
max_label_len = 0;
MatFileName = 'test.mat'
echo = false
for i = 1:length(digitStruct)
    im = imread([digitStruct(i).name]);
    min_aa=inf; min_cc=inf; max_bb=-inf; max_dd=-inf; 
    label = [];
    for j = 1:length(digitStruct(i).bbox)
        [height, width, ~] = size(im);
        aa = max(digitStruct(i).bbox(j).top+1,1);
        bb = min(digitStruct(i).bbox(j).top+digitStruct(i).bbox(j).height, height);
        cc = max(digitStruct(i).bbox(j).left+1,1);
        dd = min(digitStruct(i).bbox(j).left+digitStruct(i).bbox(j).width, width);
        
        %imshow(im(aa:bb, cc:dd, :));
        %fprintf('%d\n',digitStruct(i).bbox(j).label );
        %pause(eps);
        
        min_aa=min(min_aa, aa);
        min_cc=min(min_cc, cc);
        max_bb=max(max_bb, bb);
        max_dd=max(max_dd, dd); 
        label = [label, num2str(mod(digitStruct(i).bbox(j).label, 10))];
    end
    max_label_len = max(max_label_len, length(label));

    extend = 1/7*(6 - length(label));
    centre = 0.5*[max_bb+min_aa, max_dd+min_cc];
    half_crop_size = (1 + extend)*max(max_bb - min_aa, max_dd - min_cc)/2;

    bound_aa = max(round(centre(1) - half_crop_size), 1);
    bound_bb = min(round(centre(1) + half_crop_size), height);
    bound_cc = max(round(centre(2) - half_crop_size), 1);
    bound_dd = min(round(centre(2) + half_crop_size), width);
    scaled_im = imresize(im(bound_aa:bound_bb, bound_cc:bound_dd, :), [64,64]);
    if echo
        imshow(scaled_im);
        title(label)
        pause(1)
    end
    Images(:,:,:,i) = scaled_im;
    Labels{i} = label;
end
save(MatFileName, 'Images', 'Labels', '-v7');
max_label_len