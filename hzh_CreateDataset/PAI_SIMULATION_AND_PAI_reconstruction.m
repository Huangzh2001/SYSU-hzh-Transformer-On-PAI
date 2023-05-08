% =========================================================================
% header
% =========================================================================
clc;clear all;
%% 1.将kwave加入工具箱
% 获得当前目录
s = what;
kwave_path = fullfile(s.path,'k-wave-toolbox-version-1.4\k-Wave');
addpath(kwave_path)
%% 2.加入所有文件夹路径
% 预处理后的图像存放地址
pre_dir = '..\DataSet\Skin Cancer MNIST Pre photos\HAM10000_images_part_1';
% 光声成像重建图像存放地址
rec_dir = '..\DataSet\Skin Cancer MNIST Rec photos\HAM10000_images_part_1';
%% 3.获取想要的文件夹信息
% 读取文件夹内所有图片的路径
subdir = fullfile(pre_dir, '*.jpg');
imgdir = dir(subdir);
% 获取文件夹内所有图片个数
imgnum = length(imgdir);
% =========================================================================
% PRETREATMENT
% =========================================================================
%% 2.读取图片
for i=1:1:imgnum
    i
    % 1.读取预处理后(灰度化+归一化)的灰度图片
    I_Gray = imread(fullfile(imgdir(i).folder,imgdir(i).name));
    % imshow(I_Gray);
    
    % 2.将图像降低分辨率（降低二分之一）
    I_Gray = imresize(I_Gray,.5,'Antialiasing',false);
% =========================================================================
% SIMULATION 光声成像仿真
% =========================================================================
    % 1.将图像放大（即在外围填充一圈0）
    [row_num,col_num]=size(I_Gray);
    % a = row_num/2;
    % b = col_num/2;
    a = 5;
    b = 5;
    EnlargeNorI_Gray = padarray(I_Gray,[a b],0);
    imshow(EnlargeNorI_Gray)
    EnlargeNorI_Gray = im2double(EnlargeNorI_Gray);
    imshow(EnlargeNorI_Gray)
    size(EnlargeNorI_Gray,1)
    size(EnlargeNorI_Gray,2)

    % 2.创建计算网格
    % 读取图片的分辨率
    Nx = size(EnlargeNorI_Gray,1); % x（行）方向上的网格点数
    Ny = size(EnlargeNorI_Gray,2); % y（列）方向上的网格点数
    dx = 50e-6; % x方向上的网格点间距[m]
    dy = 50e-6; % y方向上的网格点间距[m]
    kgrid = makeGrid(Nx, dx, Ny, dy);
    
    % 3.定义介质属性
    % 人体软组织声速接近1540m/s
    %{ 
        男人人体密度估算公式：（A为上臂皮脂，B为肩胛皮脂）
        1.0913-0.0016×(A+B)
        女人人体密度估算公式：（A为上臂皮脂，B为肩胛皮脂）
        1.0897-0.00133×(A+B)
        2006年全国男人25-29岁上臂皮脂厚度监测值为10.8，肩胛皮脂监测15.8；2006年全国女人25-29岁上臂皮脂厚度监测值为17.5，肩胛皮脂监测17.5。则有：
        男人人体密度=1.0913-0.0016*(10.8+15.8)=1.0487*10**3kg/(m**3)
        女人人体密度=1.0897-0.00133*(17.5+17.5)=1.0431*10**3kg/(m**3)
    %}
    medium.sound_speed = 1540*ones(Nx, Ny); % [m/s]
    medium.density = 1048.7; % [kg/m^3]

    % 4.定义初始压力
    source.p0 = EnlargeNorI_Gray;

    % 5.定义具有50个传感器元件的中心圆的笛卡尔传感器掩模
    sensor_radius = 5.5e-3;     % [m]
    sensor_angle = 2*pi;      % [rad]
    sensor_pos = [0, 0];        % [m]
    num_sensor_points = 50;
    cart_sensor_mask = makeCartCircle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle);

    % 6.分配给传感器结构
    sensor.mask = cart_sensor_mask;
    
    % 7.设置输入选项
    input_args = {'Smooth', false, 'PMLInside', false, 'PlotPML', false};

    % 8.运行仿真
    sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
    % 仿真返回值是一个二维矩阵，大小为仿真时间点数*传感器个数，记录了每个传感器在每个时间点接收到的声压强度。
% =========================================================================
% RECONSTRUCTION 光声成像重建
% =========================================================================
    % 1.建立重构的二次计算网格
    Nx = size(EnlargeNorI_Gray,1); % x（行）方向上的网格点数
    Ny = size(EnlargeNorI_Gray,2); % y（列）方向上的网格点数
    dx = 50e-6; % x方向上的网格点间距[m]
    dy = 50e-6; % y方向上的网格点间距[m]
    kgrid_recon = kWaveGrid(Nx, dx, Ny, dy);
    
    % 2.定义介质属性
    medium.sound_speed = 1540*ones(Nx, Ny); % [m/s]
    medium.density = 1048.7; % [kg/m^3]

    % 3.使用相同的时间数组进行重建
    kgrid_recon.setTime(kgrid.Nt, kgrid.dt);

    % 4.重置初始压力
    source.p0 = 0;

    % 5.分配时间反转数据
    sensor.time_reversal_boundary_data = sensor_data;

    % 6.运行重建
    p0_recon = kspaceFirstOrder2D(kgrid_recon, medium, source, sensor, input_args{:});
% =========================================================================
% VISUALISATION AND STORE
% =========================================================================
    % 1.将图像进行截取
    I_cut = imcrop(p0_recon,[a+1 b+1 col_num-1 row_num-1]);

    % 2.将重建后的图片进行上采样以恢复分辨率
    I_resume = imresize(I_cut,2,'bilinear');

    % 3.将重建后的图片进行保存
    imwrite(I_resume,fullfile(rec_dir,[imgdir(i).name(1:end-4),'.jpg']))

    % 4.保留部分变量
    clearvars -except i imgdir imgnum kwave_path pre_dir rec_dir s sensor_data_dir subdir;
end