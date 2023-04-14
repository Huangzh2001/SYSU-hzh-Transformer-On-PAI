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
pre_dir = 'D:\Document\DataSet\Skin Cancer MNIST\Skin Cancer MNIST Pre photos\HAM10000_images_part_2';
% 仿真后的sensor_data存储地址
sensor_data_dir = 'D:\Document\DataSet\Skin Cancer MNIST\Skin Cancer MNIST sensor data\HAM10000_images_part_2';
% 光声成像仿真图像存放地址
% PAI_dir = 'D:\Document\DataSet\Skin Cancer MNIST\Skin Cancer MNIST PAI photos\HAM10000_images_part_1';
% 光声成像重建图像存放地址
rec_dir = 'D:\Document\DataSet\Skin Cancer MNIST\Skin Cancer MNIST Rec photos\HAM10000_images_part_2';
%% 3.获取想要的文件夹信息
% 读取文件夹内所有图片的路径
subdir = fullfile(pre_dir, '*.jpg');
imgdir = dir(subdir);
% 获取文件夹内所有图片个数
% imgnum = length(imgdir);
imgnum = 2000;
% =========================================================================
% PRETREATMENT
% =========================================================================
%% 2.读取图片
for i=999:1:imgnum
    i
    % 读取预处理后(灰度化+归一化)的灰度图片
    I_Gray = imread(fullfile(imgdir(i).folder,imgdir(i).name));
    % imshow(I_Gray);
    
%     % 2.将图像降低分辨率（降低二分之一）
%     image=I_Gray;
%     [M,N]=size(image);
%     %采样化后的矩阵大小
%     X=floor(M/2);
%     Y=floor(N/2);
%     x=1;y=1;
%     %采样化后的矩阵
%     A=zeros(X,Y);
%     %进行采样，每行和每列均隔三个一取
%     for m=1:2:M
%        for n=1:2:N
%           A(x,y)=image(m,n);
%           y=y+1;
%           if(y>Y)
%                 x=x+1;
%                 y=1;
%           end
%         end
%     end
%     %A为采样后的矩阵，类型是double，将其强制转化为uint8
%     A=uint8(A);
%     I_Gray=A;
%     imshow(I_Gray)
    I_Gray = imresize(I_Gray,.5,'Antialiasing',false);
% =========================================================================
% SIMULATION
% =========================================================================
    % 3.将图像放大（即在外围填充一圈0）
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

    % 3.创建计算网格
    % 读取图片的分辨率
    Nx = size(EnlargeNorI_Gray,1); % x（行）方向上的网格点数
    Ny = size(EnlargeNorI_Gray,2); % y（列）方向上的网格点数
    dx = 50e-6; % x方向上的网格点间距[m]
    dy = 50e-6; % y方向上的网格点间距[m]
    kgrid = makeGrid(Nx, dx, Ny, dy);
    
    % 4.定义介质属性
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

    % 5.使用makeTime创建时间数组，设置结束时间。使得时间节点为1600。
    %t_end = 15584e-9;
    %kgrid.t_array = makeTime(kgrid, medium.sound_speed, [], t_end);

    % 6.定义初始压力
    source.p0 = EnlargeNorI_Gray;

    % 7.定义具有30个传感器元件的中心圆的笛卡尔传感器掩模
    %sensor_radius = 7e-3; % [m]
    %num_sensor_points = 30;
    %sensor.mask = makeCartCircle(sensor_radius, num_sensor_points);
    % 定义一个居中笛卡尔圆形传感器
    sensor_radius = 5.5e-3;     % [m]
    sensor_angle = 2*pi;      % [rad]
    sensor_pos = [0, 0];        % [m]
    num_sensor_points = 50;
    cart_sensor_mask = makeCartCircle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle);

    % 分配给传感器结构
    sensor.mask = cart_sensor_mask;
    
    % 设置输入选项
    input_args = {'Smooth', false, 'PMLInside', false, 'PlotPML', false};

    % 8.运行仿真
    sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
    % 仿真返回值是一个二维矩阵，大小为仿真时间点数*传感器个数，记录了每个传感器在每个时间点接收到的声压强度。
% =========================================================================
% VISUALISATION AND STORE
% =========================================================================
    % 9.存储仿真结果
    save(fullfile(sensor_data_dir,[imgdir(i).name(1:end-4),'.mat']),'sensor_data');

    % 10.绘制模拟传感器数据
    % imagesc(sensor_data);
    % colormap(getColorMap);
    % ylabel('Sensor Position');
    % xlabel('Time Step');
    % colorbar;
    % saveas(gca,fullfile(PAI_dir,[imgdir(i).name(1:end-4),'.bmp']));
% =========================================================================
% RECONSTRUCTION
% =========================================================================
%% 3.光声成像重建
    % 1.为记录的传感器数据添加噪声
    % signal_to_noise_ratio = 10;	% [dB]
    % 为给定信噪比的信号添加高斯噪声
    % sensor_data = addNoise(sensor_data, signal_to_noise_ratio, 'peak');

    % 2.建立重构的二次计算网格
    % size(EnlargeNorI_Gray,1) = 300
    % size(EnlargeNorI_Gray,2) = 400
    Nx = size(EnlargeNorI_Gray,1); % x（行）方向上的网格点数
    Ny = size(EnlargeNorI_Gray,2); % y（列）方向上的网格点数
    dx = 50e-6; % x方向上的网格点间距[m]
    dy = 50e-6; % y方向上的网格点间距[m]
    kgrid_recon = kWaveGrid(Nx, dx, Ny, dy);
    
    % 3.定义介质属性
    medium.sound_speed = 1540*ones(Nx, Ny); % [m/s]
    medium.density = 1048.7; % [kg/m^3]

    % 4.使用相同的时间数组进行重建
    kgrid_recon.setTime(kgrid.Nt, kgrid.dt);

    % 5.重置初始压力
    source.p0 = 0;

    % 6.分配时间反转数据
    sensor.time_reversal_boundary_data = sensor_data;

    % run the time-reversal reconstruction
    p0_recon = kspaceFirstOrder2D(kgrid_recon, medium, source, sensor, input_args{:});
    
    % 7.定义具有30个传感器元件的中心圆的笛卡尔传感器掩模
    %sensor_radius = 7e-3; % [m]
    %num_sensor_points = 70;
    %sensor.mask = makeCartCircle(sensor_radius, num_sensor_points);

    % 二进制掩码
    % 创建一个等效连续圆的二进制传感器掩模 
    % sensor_radius_grid_points = round(sensor_radius / kgrid_recon.dx);
    % binary_sensor_mask = makeCircle(kgrid_recon.Nx, kgrid_recon.Ny, kgrid_recon.Nx/2 + 1, kgrid_recon.Ny/2 + 1, sensor_radius_grid_points, sensor_angle);
    
    % 分配给传感器结构
    % sensor.mask = binary_sensor_mask;
    
    % 插值数据以去除间隙并分配给传感器结构
    % interpCartData将数据从笛卡尔坐标插值到二进制传感器掩码。
    % sensor.time_reversal_boundary_data = interpCartData(kgrid_recon, sensor_data, cart_sensor_mask, binary_sensor_mask);
    
    % 8.运行时间反转重建
    % p0_recon_interp = kspaceFirstOrder2D(kgrid_recon, medium, source, sensor, input_args{:});

% =========================================================================
% VISUALISATION AND STORE
% =========================================================================
    % 1.将图像进行截取

    I_cut = imcrop(p0_recon,[a+1 b+1 col_num-1 row_num-1]);
    

    % 2.将重建后的图片进行上采样以恢复分辨率
    I_resume = imresize(I_cut,2,'bilinear');

    % 2.将重建后的图像画出
    %     % 利用插值数据绘制重建的初始压力图
    %     figure('Name', '显示图片');
    %     % imagesc(kgrid_recon.y_vec * 1e3, kgrid_recon.x_vec * 1e3, p0_recon_interp, [-1, 1]);
    %     % imagesc(kgrid_recon.y_vec * 1e3, kgrid_recon.x_vec * 1e3, p0_recon, [-1, 1]);
    %     % colormap(getColorMap);
    %     imshow(p0_recon);
    %     % 去除白边
    %     set(gca,'Position',[0 0 1 1]);
    %     % 去除x轴、y轴坐标轴刻度等
    %     set(gca,'XTick',[],'XTickLabel',[]);
    %     set(gca,'YTick',[],'YTicklabel',[]);
    %     % axis image;

    % 3.将重建后的图片进行保存
    % save(fullfile(rec_dir,[imgdir(i).name(1:end-4),'.jpg']),'p0_recon');
    imwrite(I_resume,fullfile(rec_dir,[imgdir(i).name(1:end-4),'.jpg']))

    % 保留部分变量
    clearvars -except i imgdir imgnum kwave_path pre_dir rec_dir s sensor_data_dir subdir;
end

% system('shutdown -s');