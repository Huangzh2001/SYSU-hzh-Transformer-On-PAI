close all;clear all;clc;
%% 1.批量读取数据
% 1.读取文件夹信息
% 医学图像的地址
maindir = '..\DataSet\Skin Cancer MNIST\HAM10000_images_part_1';
% 预处理后的图像存放地址
prestoredir = '..\DataSet\Skin Cancer MNIST Pre photos\HAM10000_images_part_1';

% 2.读取文件夹内所有图片的路径
subdir = fullfile(maindir, '*.jpg');
imgdir = dir(subdir);
% 3.获取文件夹内所有图片个数
imgnum = length(imgdir);

%% 2.读取图片
for i=1:1:imgnum
    i
    I_RGB = imread(fullfile(imgdir(i).folder,imgdir(i).name));

    %%
    % 1.先将图片转为灰度图
    I_Gray = 255-rgb2gray(I_RGB);
    % imshow(I_Gray)

    % 将归一化后的灰度图像储存起来
    [row_num,col_num]=size(I_Gray);
    border_width=30;
    border_mask=padarray(false(row_num-2*border_width,col_num-2*border_width),border_width,true);
    skin_mean = mean(I_Gray(border_mask));
    norI_Gray=double(I_Gray-skin_mean);
    norI_Gray=norI_Gray/max(max(abs(norI_Gray)));
    % imshow(norI_Gray,'border','tight','initialmagnification','fit');
    imwrite(norI_Gray,fullfile(prestoredir,[imgdir(i).name(1:end-4),'.jpg']))
end