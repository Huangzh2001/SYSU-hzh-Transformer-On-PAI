function sensor_data = pstdElastic3D(kgrid, medium, source, sensor, varargin)
%PSTDELASTIC3D 3D time-domain simulation of elastic wave propagation.
%
% DESCRIPTION:
%     pstdElastic3D simulates the time-domain propagation of elastic waves
%     through a three-dimensional homogeneous or heterogeneous medium given
%     four input structures: kgrid, medium, source, and sensor. The
%     computation is based on a pseudospectral time domain model which
%     accounts for viscoelastic absorption and heterogeneous material
%     parameters. At each time-step (defined by kgrid.dt and kgrid.Nt or
%     kgrid.t_array), the wavefield parameters at the positions defined by
%     sensor.mask are recorded and stored. If kgrid.t_array is set to
%     'auto', this array is automatically generated using the makeTime
%     method of the kWaveGrid class. An anisotropic absorbing boundary
%     layer called a perfectly matched layer (PML) is implemented to
%     prevent waves that leave one side of the domain being reintroduced
%     from the opposite side (a consequence of using the FFT to compute the
%     spatial derivatives in the wave equation). This allows infinite
%     domain simulations to be computed using small computational grids.
% 
%     An initial pressure distribution can be specified by assigning a
%     matrix of pressure values the same size as the computational grid to
%     source.p0. This is then assigned to the normal components of the
%     stress within the simulation function. A time varying stress source
%     can similarly be specified by assigning a binary matrix (i.e., a
%     matrix of 1's and 0's with the same dimensions as the computational
%     grid) to source.s_mask where the 1's represent the grid points that
%     form part of the source. The time varying input signals are then
%     assigned to source.sxx, source.syy, source.szz, source.sxy,
%     source.sxz, and source.syz. These can be a single time series (in
%     which case it is applied to all source elements), or a matrix of time
%     series following the source elements using MATLAB's standard
%     column-wise linear matrix index ordering. A time varying velocity
%     source can be specified in an analogous fashion, where the source
%     location is specified by source.u_mask, and the time varying input
%     velocity is assigned to source.ux, source.uy, and source.uz.
%
%     The field values are returned as arrays of time series at the sensor
%     locations defined by sensor.mask. This can be defined in three
%     different ways. (1) As a binary matrix (i.e., a matrix of 1's and 0's
%     with the same dimensions as the computational grid) representing the
%     grid points within the computational grid that will collect the data.
%     (2) As the grid coordinates of two opposing corners of a cuboid in
%     the form [x1; y1; z1; x2; y2; z2]. This is equivalent to using a
%     binary sensor mask covering the same region, however, the output is
%     indexed differently as discussed below. (3) As a series of Cartesian
%     coordinates within the grid which specify the location of the
%     pressure values stored at each time step. If the Cartesian
%     coordinates don't exactly match the coordinates of a grid point, the
%     output values are calculated via interpolation. The Cartesian points
%     must be given as a 3 by N matrix corresponding to the x, y, and z
%     positions, respectively, where the Cartesian origin is assumed to be
%     in the center of the grid. If no output is required, the sensor input
%     can be replaced with an empty array [].
%
%     If sensor.mask is given as a set of Cartesian coordinates, the
%     computed sensor_data is returned in the same order. If sensor.mask is
%     given as a binary matrix, sensor_data is returned using MATLAB's
%     standard column-wise linear matrix index ordering. In both cases, the
%     recorded data is indexed as sensor_data(sensor_point_index,
%     time_index). For a binary sensor mask, the field values at a
%     particular time can be restored to the sensor positions within the
%     computation grid using unmaskSensorData. If sensor.mask is given as a
%     list of cuboid corners, the recorded data is indexed as
%     sensor_data(cuboid_index).p(x_index, y_index, z_index, time_index),
%     where x_index, y_index, and z_index correspond to the grid index
%     within the cuboid, and cuboid_index corresponds to the number of the
%     cuboid if more than one is specified. 
%
%     By default, the recorded acoustic pressure field is passed directly
%     to the output sensor_data. However, other acoustic parameters can
%     also be recorded by setting sensor.record to a cell array of the form
%     {'p', 'u', 'p_max', ...}. For example, both the particle velocity and
%     the acoustic pressure can be returned by setting sensor.record =
%     {'p', 'u'}. If sensor.record is given, the output sensor_data is
%     returned as a structure with the different outputs appended as
%     structure fields. For example, if sensor.record = {'p', 'p_final',
%     'p_max', 'u'}, the output would contain fields sensor_data.p,
%     sensor_data.p_final, sensor_data.p_max, sensor_data.ux,
%     sensor_data.uy, and sensor_data.uz. Most of the output parameters are
%     recorded at the given sensor positions and are indexed as
%     sensor_data.field(sensor_point_index, time_index) or
%     sensor_data(cuboid_index).field(x_index, y_index, z_index,
%     time_index) if using a sensor mask defined as cuboid corners. The
%     exceptions are the averaged quantities ('p_max', 'p_rms', 'u_max',
%     'p_rms', 'I_avg'), the 'all' quantities ('p_max_all', 'p_min_all',
%     'u_max_all', 'u_min_all'), and the final quantities ('p_final',
%     'u_final'). The averaged quantities are indexed as
%     sensor_data.p_max(sensor_point_index) or
%     sensor_data(cuboid_index).p_max(x_index, y_index, z_index) if using
%     cuboid corners, while the final and 'all' quantities are returned
%     over the entire grid and are always indexed as
%     sensor_data.p_final(nx, ny, nz), regardless of the type of sensor
%     mask.
%
%     pstdElastic3D may also be used for time reversal image reconstruction
%     by assigning the time varying pressure recorded over an arbitrary
%     sensor surface to the input field sensor.time_reversal_boundary_data.
%     This data is then enforced in time reversed order as a time varying
%     Dirichlet boundary condition over the sensor surface given by
%     sensor.mask. The boundary data must be indexed as
%     sensor.time_reversal_boundary_data(sensor_point_index, time_index).
%     If sensor.mask is given as a set of Cartesian coordinates, the
%     boundary data must be given in the same order. An equivalent binary
%     sensor mask (computed using nearest neighbour interpolation) is then
%     used to place the pressure values into the computational grid at each
%     time step. If sensor.mask is given as a binary matrix of sensor
%     points, the boundary data must be ordered using MATLAB's standard
%     column-wise linear matrix indexing. If no additional inputs are
%     required, the source input can be replaced with an empty array [].
%
% USAGE:
%     sensor_data = pstdElastic3D(kgrid, medium, source, sensor)
%     sensor_data = pstdElastic3D(kgrid, medium, source, sensor, ...) 
%
% INPUTS:
% The minimum fields that must be assigned to run an initial value problem
% (for example, a photoacoustic forward simulation) are marked with a *. 
%
%     kgrid*                 - k-Wave grid object returned by kWaveGrid
%                              containing Cartesian and k-space grid fields 
%     kgrid.t_array*         - evenly spaced array of time values [s] (set
%                              to 'auto' by kWaveGrid) 
%
%     medium.sound_speed_compression*
%                            - compressional sound speed distribution
%                              within the acoustic medium [m/s] 
%     medium.sound_speed_shear*
%                            - shear sound speed distribution within the
%                              acoustic medium [m/s] 
%     medium.density*        - density distribution within the acoustic
%                              medium [kg/m^3] 
%     medium.alpha_coeff_compression
%                            - absorption coefficient for compressional
%                              waves [dB/(MHz^2 cm)] 
%     medium.alpha_coeff_shear
%                            - absorption coefficient for shear waves
%                              [dB/(MHz^2 cm)] 
%
%     source.p0*             - initial pressure within the acoustic medium
%     source.sxx             - time varying stress at each of the source
%                              positions given by source.s_mask 
%     source.syy             - time varying stress at each of the source
%                              positions given by source.s_mask 
%     source.szz             - time varying stress at each of the source
%                              positions given by source.s_mask 
%     source.sxy             - time varying stress at each of the source
%                              positions given by source.s_mask 
%     source.sxz             - time varying stress at each of the source
%                              positions given by source.s_mask 
%     source.syz             - time varying stress at each of the source
%                              positions given by source.s_mask 
%     source.s_mask          - binary matrix specifying the positions of
%                              the time varying stress source distributions 
%     source.s_mode          - optional input to control whether the input
%                              stress is injected as a mass source or
%                              enforced as a dirichlet boundary condition;
%                              valid inputs are 'additive' (the default) or
%                              'dirichlet'    
%     source.ux              - time varying particle velocity in the
%                              x-direction at each of the source positions
%                              given by source.u_mask  
%     source.uy              - time varying particle velocity in the
%                              y-direction at each of the source positions
%                              given by source.u_mask  
%     source.uz              - time varying particle velocity in the
%                              z-direction at each of the source positions
%                              given by source.u_mask  
%     source.u_mask          - binary matrix specifying the positions of
%                              the time varying particle velocity
%                              distribution  
%     source.u_mode          - optional input to control whether the input
%                              velocity is applied as a force source or
%                              enforced as a dirichlet boundary condition;
%                              valid inputs are 'additive' (the default) or
%                              'dirichlet'    
%
%     sensor.mask*           - binary matrix or a set of Cartesian points
%                              where the pressure is recorded at each
%                              time-step  
%     sensor.record          - cell array of the acoustic parameters to
%                              record in the form sensor.record = {'p',
%                              'u', ...}; valid inputs are:
%
%         'p' (acoustic pressure)
%         'p_max' (maximum pressure)
%         'p_min' (minimum pressure)
%         'p_rms' (RMS pressure)
%         'p_final' (final pressure field at all grid points)
%         'p_max_all' (maximum pressure at all grid points)
%         'p_min_all' (minimum pressure at all grid points)
%         'u' (particle velocity)
%         'u_max' (maximum particle velocity)
%         'u_min' (minimum particle velocity)
%         'u_rms' (RMS particle21st January 2014 velocity)
%         'u_final' (final particle velocity field at all grid points)
%         'u_max_all' (maximum particle velocity at all grid points)
%         'u_min_all' (minimum particle velocity at all grid points)
%         'u_non_staggered' (particle velocity on non-staggered grid)
%         'u_split_field' (particle velocity on non-staggered grid split
%                          into compressional and shear components)   
%         'I' (time varying acoustic intensity)
%         'I_avg' (average acoustic intensity)
%
%         NOTE: the acoustic pressure outputs are calculated from the
%         normal stress via: p = -(sxx + syy)/2  
%
%     sensor.record_start_index
%                            - time index at which the sensor should start
%                              recording the data specified by
%                              sensor.record (default = 1)  
%     sensor.time_reversal_boundary_data
%                            - time varying pressure enforced as a
%                              Dirichlet boundary condition over
%                              sensor.mask
%
% Note: For a heterogeneous medium, medium.sound_speed_compression,
% medium.sound_speed_shear, and medium.density must be given in matrix form
% with the same dimensions as kgrid. For a homogeneous medium, these can be
% given as scalar values.   
%
% OPTIONAL INPUTS:
%     Optional 'string', value pairs that may be used to modify the default
%     computational settings.
%
%     See .html help file for details.
%
% OUTPUTS:
% If sensor.record is not defined by the user:
%     sensor_data            - time varying pressure recorded at the sensor
%                              positions given by sensor.mask
%
% If sensor.record is defined by the user:
%     sensor_data.p          - time varying pressure recorded at the
%                              sensor positions given by sensor.mask
%                              (returned if 'p' is set)  
%     sensor_data.p_max      - maximum pressure recorded at the sensor
%                              positions given by sensor.mask (returned if
%                              'p_max' is set)  
%     sensor_data.p_min      - minimum pressure recorded at the sensor
%                              positions given by sensor.mask (returned if
%                              'p_min' is set)  
%     sensor_data.p_rms      - rms of the time varying pressure recorded
%                              at the sensor positions given by
%                              sensor.mask (returned if 'p_rms' is set)  
%     sensor_data.p_final    - final pressure field at all grid points
%                              within the domain (returned if 'p_final' is
%                              set)
%     sensor_data.p_max_all  - maximum pressure recorded at all grid points
%                              within the domain (returned if 'p_max_all'
%                              is set)  
%     sensor_data.p_min_all  - minimum pressure recorded at all grid points
%                              within the domain (returned if 'p_min_all'
%                              is set)
%     sensor_data.ux         - time varying particle velocity in the
%                              x-direction recorded at the sensor positions
%                              given by sensor.mask (returned if 'u' is
%                              set)  
%     sensor_data.uy         - time varying particle velocity in the
%                              y-direction recorded at the sensor positions
%                              given by sensor.mask (returned if 'u' is
%                              set)
%     sensor_data.uz         - time varying particle velocity in the
%                              z-direction recorded at the sensor positions
%                              given by sensor.mask (returned if 'u' is
%                              set)
%     sensor_data.ux_max     - maximum particle velocity in the x-direction
%                              recorded at the sensor positions given by
%                              sensor.mask (returned if 'u_max' is set)   
%     sensor_data.uy_max     - maximum particle velocity in the y-direction
%                              recorded at the sensor positions given by
%                              sensor.mask (returned if 'u_max' is set)   
%     sensor_data.uz_max     - maximum particle velocity in the z-direction
%                              recorded at the sensor positions given by
%                              sensor.mask (returned if 'u_max' is set)  
%     sensor_data.ux_min     - minimum particle velocity in the x-direction
%                              recorded at the sensor positions given by
%                              sensor.mask (returned if 'u_min' is set) 
%     sensor_data.uy_min     - minimum particle velocity in the y-direction
%                              recorded at the sensor positions given by
%                              sensor.mask (returned if 'u_min' is set)   
%     sensor_data.uz_min     - minimum particle velocity in the z-direction
%                              recorded at the sensor positions given by
%                              sensor.mask (returned if 'u_min' is set) 
%     sensor_data.ux_rms     - rms of the time varying particle velocity in
%                              the x-direction recorded at the sensor
%                              positions given by sensor.mask (returned if
%                              'u_rms' is set)   
%     sensor_data.uy_rms     - rms of the time varying particle velocity in
%                              the y-direction recorded at the sensor
%                               positions given by sensor.mask (returned if
%                              'u_rms' is set)   
%     sensor_data.uz_rms     - rms of the time varying particle velocity
%                              in the z-direction recorded at the sensor
%                              positions given by sensor.mask (returned if
%                               'u_rms' is set)   
%     sensor_data.ux_final   - final particle velocity field in the
%                              x-direction at all grid points within the
%                              domain (returned if 'u_final' is set) 
%     sensor_data.uy_final   - final particle velocity field in the
%                              y-direction at all grid points within the
%                              domain (returned if 'u_final' is set) 
%     sensor_data.uz_final   - final particle velocity field in the
%                              z-direction at all grid points within the
%                              domain (returned if 'u_final' is set) 
%     sensor_data.ux_max_all - maximum particle velocity in the x-direction
%                              recorded at all grid points within the
%                              domain (returned if 'u_max_all' is set)   
%     sensor_data.uy_max_all - maximum particle velocity in the y-direction
%                              recorded at all grid points within the
%                              domain (returned if 'u_max_all' is set)   
%     sensor_data.uz_max_all - maximum particle velocity in the z-direction
%                              recorded at all grid points within the
%                              domain (returned if 'u_max_all' is set)  
%     sensor_data.ux_min_all - minimum particle velocity in the x-direction
%                              recorded at all grid points within the
%                              domain (returned if 'u_min_all' is set)   
%     sensor_data.uy_min_all - minimum particle velocity in the y-direction
%                              recorded at all grid points within the
%                              domain (returned if 'u_min_all' is set)   
%     sensor_data.uz_min_all - minimum particle velocity in the z-direction
%                              recorded at all grid points within the
%                              domain (returned if 'u_min_all' is set) 
%     sensor_data.ux_non_staggered 
%                            - time varying particle velocity in the
%                              x-direction recorded at the sensor positions
%                              given by sensor.mask after shifting to the
%                              non-staggered grid (returned if
%                              'u_non_staggered' is set)  
%     sensor_data.uy_non_staggered 
%                            - time varying particle velocity in the
%                              y-direction recorded at the sensor positions
%                              given by sensor.mask after shifting to the
%                              non-staggered grid (returned if
%                              'u_non_staggered' is set)  
%     sensor_data.uz_non_staggered 
%                            - time varying particle velocity in the
%                              z-direction recorded at the sensor positions
%                              given by sensor.mask after shifting to the
%                              non-staggered grid (returned if
%                              'u_non_staggered' is set)
%     sensor_data.ux_split_p - compressional component of the time varying
%                              particle velocity in the x-direction on the
%                              non-staggered grid recorded at the sensor
%                              positions given by sensor.mask (returned if
%                              'u_split_field' is set)
%     sensor_data.ux_split_s - shear component of the time varying particle
%                              velocity in the x-direction on the 
%                              non-staggered grid recorded at the sensor
%                              positions given by sensor.mask (returned if
%                              'u_split_field' is set)
%     sensor_data.uy_split_p - compressional component of the time varying
%                              particle velocity in the y-direction on the
%                              non-staggered grid recorded at the sensor
%                              positions given by sensor.mask (returned if
%                              'u_split_field' is set)
%     sensor_data.uy_split_s - shear component of the time varying particle
%                              velocity in the y-direction on the 
%                              non-staggered grid recorded at the sensor
%                              positions given by sensor.mask (returned if
%                              'u_split_field' is set)
%     sensor_data.uz_split_p - compressional component of the time varying
%                              particle velocity in the z-direction on the
%                              non-staggered grid recorded at the sensor
%                              positions given by sensor.mask (returned if
%                              'u_split_field' is set)
%     sensor_data.uz_split_s - shear component of the time varying particle
%                              velocity in the z-direction on the 
%                              non-staggered grid recorded at the sensor
%                              positions given by sensor.mask (returned if
%                              'u_split_field' is set)
%     sensor_data.Ix         - time varying acoustic intensity in the
%                              x-direction recorded at the sensor positions
%                              given by sensor.mask (returned if 'I' is
%                              set)  
%     sensor_data.Iy         - time varying acoustic intensity in the
%                              y-direction recorded at the sensor positions
%                              given by sensor.mask (returned if 'I' is
%                              set)  
%     sensor_data.Iz         - time varying acoustic intensity in the
%                              z-direction recorded at the sensor positions
%                              given by sensor.mask (returned if 'I' is
%                              set)  
%     sensor_data.Ix_avg     - average acoustic intensity in the
%                              x-direction recorded at the sensor positions
%                              given by sensor.mask (returned if 'I_avg' is
%                              set)
%     sensor_data.Iy_avg     - average acoustic intensity in the
%                              y-direction recorded at the sensor positions
%                              given by sensor.mask (returned if 'I_avg' is
%                              set)
%     sensor_data.Iz_avg     - average acoustic intensity in the
%                              z-direction recorded at the sensor positions
%                              given by sensor.mask (returned if 'I_avg' is
%                              set)
%
% ABOUT:
%     author                 - Bradley Treeby & Ben Cox
%     date                   - 11th March 2013
%     last update            - 13th January 2019
%
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2013-2019 Bradley Treeby and Ben Cox
%
% See also kspaceFirstOrder3D, kWaveGrid, pstdElastic2D

% This file is part of k-Wave. k-Wave is free software: you can
% redistribute it and/or modify it under the terms of the GNU Lesser
% General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% 
% k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
% more details. 
% 
% You should have received a copy of the GNU Lesser General Public License
% along with k-Wave. If not, see <http://www.gnu.org/licenses/>.

% suppress mlint warnings that arise from using subscripts
%#ok<*NASGU>
%#ok<*COLND>
%#ok<*NODEF>
%#ok<*INUSL>

% =========================================================================
% CHECK INPUT STRUCTURES AND OPTIONAL INPUTS
% =========================================================================

% start the timer and store the start time
start_time = clock;
tic;

% set the name of the simulation code
MFILE = mfilename;

% get the number of inputs and outputs (nargin and nargout can't be used in
% subscripts in MATLAB 2016b or later)
num_inputs  = nargin;
num_outputs = nargout;

% run subscript to check inputs
kspaceFirstOrder_inputChecking;

% assign the lame parameters
mu     = medium.sound_speed_shear.^2       .* medium.density;
lambda = medium.sound_speed_compression.^2 .* medium.density - 2 .* mu;

% assign the viscosity coefficients
if flags.kelvin_voigt_model
    eta = 2 * rho0 .* medium.sound_speed_shear .^ 3      .* db2neper(medium.alpha_coeff_shear, 2);
    chi = 2 * rho0 .* medium.sound_speed_compression .^3 .* db2neper(medium.alpha_coeff_compression, 2) - 2 * eta;
end

% =========================================================================
% CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
% =========================================================================

% calculate the values of the density at the staggered grid points
% using the arithmetic average [1, 2], where sgx  = (x + dx/2, y), 
% sgy  = (x, y + dy/2) and sgz = (x, y, z + dz/2)
if numDim(rho0) == 3 && flags.use_sg
    
    % rho0 is heterogeneous and staggered grids are used
    rho0_sgx = interpn(kgrid.x, kgrid.y, kgrid.z, rho0, kgrid.x + kgrid.dx/2, kgrid.y, kgrid.z, '*linear');
    rho0_sgy = interpn(kgrid.x, kgrid.y, kgrid.z, rho0, kgrid.x, kgrid.y + kgrid.dy/2, kgrid.z, '*linear');
    rho0_sgz = interpn(kgrid.x, kgrid.y, kgrid.z, rho0, kgrid.x, kgrid.y, kgrid.z + kgrid.dz/2, '*linear');
        
    % set values outside of the interpolation range to original values
    rho0_sgx(isnan(rho0_sgx)) = rho0(isnan(rho0_sgx));
    rho0_sgy(isnan(rho0_sgy)) = rho0(isnan(rho0_sgy));    
    rho0_sgz(isnan(rho0_sgz)) = rho0(isnan(rho0_sgz));
    
else
    
    % rho0 is homogeneous or staggered grids are not used
    rho0_sgx = rho0;
    rho0_sgy = rho0;
    rho0_sgz = rho0; 
    
end

% invert rho0 so it doesn't have to be done each time step
rho0_sgx_inv = 1./rho0_sgx;
rho0_sgy_inv = 1./rho0_sgy;
rho0_sgz_inv = 1./rho0_sgz;

% clear unused variables if not using them in _saveToDisk
if ~flags.save_to_disk
    clear rho0_sgx rho0_sgy rho0_sgz
end

% calculate the values of mu at the staggered grid points using the
% harmonic average [1, 2], where sgxy = (x + dx/2, y + dy/2, z), etc
if numDim(mu) == 3 && flags.use_sg
    
    % mu is heterogeneous and staggered grids are used
    mu_sgxy  = 1./interpn(kgrid.x, kgrid.y, kgrid.z, 1./mu, kgrid.x + kgrid.dx/2, kgrid.y + kgrid.dy/2, kgrid.z, '*linear');
    mu_sgxz  = 1./interpn(kgrid.x, kgrid.y, kgrid.z, 1./mu, kgrid.x + kgrid.dx/2, kgrid.y, kgrid.z + kgrid.dz/2, '*linear');
    mu_sgyz  = 1./interpn(kgrid.x, kgrid.y, kgrid.z, 1./mu, kgrid.x, kgrid.y + kgrid.dy/2, kgrid.z + kgrid.dz/2, '*linear');
        
    % set values outside of the interpolation range to original values
    mu_sgxy(isnan(mu_sgxy)) = mu(isnan(mu_sgxy));
    mu_sgxz(isnan(mu_sgxz)) = mu(isnan(mu_sgxz));
    mu_sgyz(isnan(mu_sgyz)) = mu(isnan(mu_sgyz));    
    
else
    
    % mu is homogeneous or staggered grids are not used
    mu_sgxy  = mu;
    mu_sgxz  = mu;
    mu_sgyz  = mu; 
    
end

% calculate the values of eta at the staggered grid points using the
% harmonic average [1, 2], where sgxy = (x + dx/2, y + dy/2, z) etc 
if flags.kelvin_voigt_model
    if numDim(eta) == 3 && flags.use_sg
    
        % eta is heterogeneous and staggered grids are used
        eta_sgxy  = 1./interpn(kgrid.x, kgrid.y, kgrid.z, 1./eta, kgrid.x + kgrid.dx/2, kgrid.y + kgrid.dy/2, kgrid.z, '*linear');
        eta_sgxz  = 1./interpn(kgrid.x, kgrid.y, kgrid.z, 1./eta, kgrid.x + kgrid.dx/2, kgrid.y, kgrid.z + kgrid.dz/2, '*linear');
        eta_sgyz  = 1./interpn(kgrid.x, kgrid.y, kgrid.z, 1./eta, kgrid.x, kgrid.y + kgrid.dy/2, kgrid.z + kgrid.dz/2, '*linear');

        % set values outside of the interpolation range to original values 
        eta_sgxy(isnan(eta_sgxy)) = eta(isnan(eta_sgxy));
        eta_sgxz(isnan(eta_sgxz)) = eta(isnan(eta_sgxz));
        eta_sgyz(isnan(eta_sgyz)) = eta(isnan(eta_sgyz));   
    
    else
        
        % eta is homogeneous or staggered grids are not used
        eta_sgxy = eta;
        eta_sgxz = eta;
        eta_sgyz = eta;
        
    end
end

% [1] Moczo, P., Kristek, J., Vavry?uk, V., Archuleta, R. J., & Halada, L.
% (2002). 3D heterogeneous staggered-grid finite-difference modeling of
% seismic motion with volume harmonic and arithmetic averaging of elastic
% moduli and densities. Bulletin of the Seismological Society of America,
% 92(8), 3042�3066.    

% [2] Toyoda, M., Takahashi, D., & Kawai, Y. (2012). Averaged material
% parameters and boundary conditions for the vibroacoustic
% finite-difference time-domain method with a nonuniform mesh. Acoustical
% Science and Technology, 33(4), 273�276.  

% =========================================================================
% PREPARE DERIVATIVE AND PML OPERATORS
% =========================================================================

% get the regular PML operators based on the reference sound speed and PML settings
pml_x     = getPML(kgrid.Nx, kgrid.dx, kgrid.dt, c_ref, pml_x_size, pml_x_alpha, false, 1);
pml_x_sgx = getPML(kgrid.Nx, kgrid.dx, kgrid.dt, c_ref, pml_x_size, pml_x_alpha, true && flags.use_sg, 1);
pml_y     = getPML(kgrid.Ny, kgrid.dy, kgrid.dt, c_ref, pml_y_size, pml_y_alpha, false, 2);
pml_y_sgy = getPML(kgrid.Ny, kgrid.dy, kgrid.dt, c_ref, pml_y_size, pml_y_alpha, true && flags.use_sg, 2);
pml_z     = getPML(kgrid.Nz, kgrid.dz, kgrid.dt, c_ref, pml_z_size, pml_z_alpha, false, 3);
pml_z_sgz = getPML(kgrid.Nz, kgrid.dz, kgrid.dt, c_ref, pml_z_size, pml_z_alpha, true && flags.use_sg, 3);

% get the multi-axial PML operators
mpml_x     = getPML(kgrid.Nx, kgrid.dx, kgrid.dt, c_ref, pml_x_size, multi_axial_PML_ratio * pml_x_alpha, false, 1);
mpml_x_sgx = getPML(kgrid.Nx, kgrid.dx, kgrid.dt, c_ref, pml_x_size, multi_axial_PML_ratio * pml_x_alpha, true && flags.use_sg, 1);
mpml_y     = getPML(kgrid.Ny, kgrid.dy, kgrid.dt, c_ref, pml_y_size, multi_axial_PML_ratio * pml_y_alpha, false, 2);
mpml_y_sgy = getPML(kgrid.Ny, kgrid.dy, kgrid.dt, c_ref, pml_y_size, multi_axial_PML_ratio * pml_y_alpha, true && flags.use_sg, 2);
mpml_z     = getPML(kgrid.Nz, kgrid.dz, kgrid.dt, c_ref, pml_z_size, multi_axial_PML_ratio * pml_z_alpha, false, 3);
mpml_z_sgz = getPML(kgrid.Nz, kgrid.dz, kgrid.dt, c_ref, pml_z_size, multi_axial_PML_ratio * pml_z_alpha, true && flags.use_sg, 3);

% define the k-space derivative operators, multiply by the staggered
% grid shift operators, and then re-order using ifftshift (the option
% flags.use_sg exists for debugging) 
if flags.use_sg
    ddx_k_shift_pos = ifftshift( 1i * kgrid.kx_vec .* exp( 1i * kgrid.kx_vec * kgrid.dx / 2) );
    ddx_k_shift_neg = ifftshift( 1i * kgrid.kx_vec .* exp(-1i * kgrid.kx_vec * kgrid.dx / 2) );
    ddy_k_shift_pos = ifftshift( 1i * kgrid.ky_vec .* exp( 1i * kgrid.ky_vec * kgrid.dy / 2) );
    ddy_k_shift_neg = ifftshift( 1i * kgrid.ky_vec .* exp(-1i * kgrid.ky_vec * kgrid.dy / 2) );
    ddz_k_shift_pos = ifftshift( 1i * kgrid.kz_vec .* exp( 1i * kgrid.kz_vec * kgrid.dz / 2) );
    ddz_k_shift_neg = ifftshift( 1i * kgrid.kz_vec .* exp(-1i * kgrid.kz_vec * kgrid.dz / 2) );
else
    ddx_k_shift_pos = ifftshift( 1i * kgrid.kx_vec );
    ddx_k_shift_neg = ifftshift( 1i * kgrid.kx_vec );
    ddy_k_shift_pos = ifftshift( 1i * kgrid.ky_vec );
    ddy_k_shift_neg = ifftshift( 1i * kgrid.ky_vec );
    ddz_k_shift_pos = ifftshift( 1i * kgrid.kz_vec );
    ddz_k_shift_neg = ifftshift( 1i * kgrid.kz_vec );         
end

% force the derivative and shift operators to be in the correct direction
% for use with BSXFUN
ddy_k_shift_pos = ddy_k_shift_pos.'; 
ddy_k_shift_neg = ddy_k_shift_neg.';
ddz_k_shift_pos = permute(ddz_k_shift_pos, [2, 3, 1]);
ddz_k_shift_neg = permute(ddz_k_shift_neg, [2, 3, 1]);

% =========================================================================
% SAVE DATA TO DISK FOR RUNNING SIMULATION EXTERNAL TO MATLAB
% =========================================================================

% save to disk option for saving the input matrices to disk for running
% simulations using k-Wave++
if flags.save_to_disk
    
    % run subscript to save files to disk
    kspaceFirstOrder_saveToDisk;
    
    % run subscript to resize the transducer object if the grid has been
    % expanded 
    kspaceFirstOrder_retractTransducerGridSize;
    
    % exit matlab computation if required
    if flags.save_to_disk_exit
        return
    end
    
end

% =========================================================================
% DATA CASTING
% =========================================================================

% preallocate the loop variables using the castZeros anonymous function
% (this creates a matrix of zeros in the data type specified by data_cast)
ux_split_x  = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
ux_split_y  = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
ux_split_z  = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
ux_sgx      = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
uy_split_x  = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
uy_split_y  = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
uy_split_z  = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
uy_sgy      = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
uz_split_x  = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
uz_split_y  = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
uz_split_z  = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
uz_sgz      = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **

sxx_split_x = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
sxx_split_y = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
sxx_split_z = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
syy_split_x = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
syy_split_y = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
syy_split_z = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
szz_split_x = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
szz_split_y = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
szz_split_z = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
sxy_split_x = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
sxy_split_y = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
sxz_split_x = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
sxz_split_z = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
syz_split_y = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
syz_split_z = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);

duxdx       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
duxdy       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
duxdz       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
duydx       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
duydy       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
duydz       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
duzdx       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
duzdy       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
duzdz       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **

dsxxdx      = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
dsyydy      = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
dszzdz      = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
dsxydx      = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
dsxydy      = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
dsxzdx      = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
dsxzdz      = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
dsyzdy      = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
dsyzdz      = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **

p           = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **

if flags.kelvin_voigt_model
    dduxdxdt       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
    dduxdydt       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
    dduxdzdt       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
    dduydxdt       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
    dduydydt       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
    dduydzdt       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
    dduzdxdt       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
    dduzdydt       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
    dduzdzdt       = castZeros([kgrid.Nx, kgrid.Ny, kgrid.Nz]);  % **
end

% to save memory, the variables noted with a ** do not neccesarily need to
% be explicitly stored (they are not needed for update steps). Instead they
% could be replaced with a small number of temporary variables that are
% reused several times during the time loop.

% run subscript to cast the remaining loop variables to the data type
% specified by data_cast 
if ~strcmp(data_cast, 'off')
    kspaceFirstOrder_dataCast;
end

% =========================================================================
% CREATE INDEX VARIABLES
% =========================================================================

% setup the time index variable
if ~flags.time_rev
    index_start = 1;
    index_step = 1;
    index_end = kgrid.Nt; 
else
    
    % throw error for unsupported feature
    error('Time reversal using sensor.time_reversal_boundary_data is not currently supported.');
        
end

% =========================================================================
% PREPARE VISUALISATIONS
% =========================================================================

% pre-compute suitable axes scaling factor
if flags.plot_layout || flags.plot_sim
    [x_sc, scale, prefix] = scaleSI(max([kgrid.x_vec; kgrid.y_vec; kgrid.z_vec])); %#ok<ASGLU>
end

% throw error for currently unsupported plot layout feature
if flags.plot_layout
    error('''PlotLayout'' input is not currently supported.');
end

% initialise the figure used for animation if 'PlotSim' is set to 'true'
if flags.plot_sim
    kspaceFirstOrder_initialiseFigureWindow;
end 

% initialise movie parameters if 'RecordMovie' is set to 'true'
if flags.record_movie
    kspaceFirstOrder_initialiseMovieParameters;
end

% =========================================================================
% LOOP THROUGH TIME STEPS
% =========================================================================

% update command line status
disp(['  precomputation completed in ' scaleTime(toc)]);
disp('  starting time loop...');

% restart timing variables
loop_start_time = clock;
tic;

% start time loop
for t_index = index_start:index_step:index_end

    % compute the gradients of the stress tensor (these variables do not
    % necessaily need to be stored, they could be computed as needed)
    dsxxdx = real( ifft( bsxfun(@times, ddx_k_shift_pos, fft(sxx_split_x + sxx_split_y + sxx_split_z, [], 1)), [], 1) );
    dsyydy = real( ifft( bsxfun(@times, ddy_k_shift_pos, fft(syy_split_x + syy_split_y + syy_split_z, [], 2)), [], 2) );
    dszzdz = real( ifft( bsxfun(@times, ddz_k_shift_pos, fft(szz_split_x + szz_split_y + szz_split_z, [], 3)), [], 3) );
    dsxydx = real( ifft( bsxfun(@times, ddx_k_shift_neg, fft(sxy_split_x + sxy_split_y, [], 1)), [], 1) );
    dsxydy = real( ifft( bsxfun(@times, ddy_k_shift_neg, fft(sxy_split_x + sxy_split_y, [], 2)), [], 2) );
    dsxzdx = real( ifft( bsxfun(@times, ddx_k_shift_neg, fft(sxz_split_x + sxz_split_z, [], 1)), [], 1) );
    dsxzdz = real( ifft( bsxfun(@times, ddz_k_shift_neg, fft(sxz_split_x + sxz_split_z, [], 3)), [], 3) );
    dsyzdy = real( ifft( bsxfun(@times, ddy_k_shift_neg, fft(syz_split_y + syz_split_z, [], 2)), [], 2) );
    dsyzdz = real( ifft( bsxfun(@times, ddz_k_shift_neg, fft(syz_split_y + syz_split_z, [], 3)), [], 3) );

    % calculate the split-field components of ux_sgx, uy_sgy, and uz_sgz at
    % the next time step using the components of the stress at the current
    % time step 
    ux_split_x = bsxfun(@times, mpml_z,     bsxfun(@times, mpml_y,     bsxfun(@times, pml_x_sgx, ...
                 bsxfun(@times, mpml_z,     bsxfun(@times, mpml_y,     bsxfun(@times, pml_x_sgx, ux_split_x))) ...
                 + dt .* rho0_sgx_inv .* dsxxdx)));
    ux_split_y = bsxfun(@times, mpml_x_sgx, bsxfun(@times, mpml_z,     bsxfun(@times, pml_y, ...
                 bsxfun(@times, mpml_x_sgx, bsxfun(@times, mpml_z,     bsxfun(@times, pml_y, ux_split_y))) ...
                 + dt .* rho0_sgx_inv .* dsxydy)));  
    ux_split_z = bsxfun(@times, mpml_y,     bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z, ...
                 bsxfun(@times, mpml_y,     bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z, ux_split_z))) ...
                 + dt .* rho0_sgx_inv .* dsxzdz)));
              
    uy_split_x = bsxfun(@times, mpml_z,     bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x, ...
                 bsxfun(@times, mpml_z,     bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x, uy_split_x))) ...
                 + dt .* rho0_sgy_inv .* dsxydx)));
    uy_split_y = bsxfun(@times, mpml_x,     bsxfun(@times, mpml_z,     bsxfun(@times, pml_y_sgy, ...
                 bsxfun(@times, mpml_x,     bsxfun(@times, mpml_z,     bsxfun(@times, pml_y_sgy, uy_split_y))) ...
                 + dt .* rho0_sgy_inv .* dsyydy)));
    uy_split_z = bsxfun(@times, mpml_y_sgy, bsxfun(@times, mpml_x,     bsxfun(@times, pml_z, ...
                 bsxfun(@times, mpml_y_sgy, bsxfun(@times, mpml_x,     bsxfun(@times, pml_z, uy_split_z))) ...
                 + dt .* rho0_sgy_inv .* dsyzdz)));
               
    uz_split_x = bsxfun(@times, mpml_z_sgz, bsxfun(@times, mpml_y,     bsxfun(@times, pml_x, ...
                 bsxfun(@times, mpml_z_sgz, bsxfun(@times, mpml_y,     bsxfun(@times, pml_x, uz_split_x))) ...
                 + dt .* rho0_sgz_inv .* dsxzdx)));
    uz_split_y = bsxfun(@times, mpml_x,     bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y, ...
                 bsxfun(@times, mpml_x,     bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y, uz_split_y))) ...
                 + dt .* rho0_sgz_inv .* dsyzdy)));
    uz_split_z = bsxfun(@times, mpml_y,     bsxfun(@times, mpml_x,     bsxfun(@times, pml_z_sgz, ...
                 bsxfun(@times, mpml_y,     bsxfun(@times, mpml_x,     bsxfun(@times, pml_z_sgz, uz_split_z))) ...
                 + dt .* rho0_sgz_inv .* dszzdz)));

    % add in the velocity source terms
    if flags.source_ux >= t_index
        if strcmp(source.u_mode, 'dirichlet')
            
            % enforce the source values as a dirichlet boundary condition
            ux_split_x(u_source_pos_index) = source.ux(u_source_sig_index, t_index);
            
        else
            
            % add the source values to the existing field values 
            ux_split_x(u_source_pos_index) = ux_split_x(u_source_pos_index) + source.ux(u_source_sig_index, t_index);
            
        end
    end
    if flags.source_uy >= t_index
        if strcmp(source.u_mode, 'dirichlet')
            
            % enforce the source values as a dirichlet boundary condition        
            uy_split_y(u_source_pos_index) = source.uy(u_source_sig_index, t_index);
            
        else
            
            % add the source values to the existing field values 
            uy_split_y(u_source_pos_index) = uy_split_y(u_source_pos_index) + source.uy(u_source_sig_index, t_index);
            
        end
    end    
    if flags.source_uz >= t_index
        if strcmp(source.u_mode, 'dirichlet')
            
            % enforce the source values as a dirichlet boundary condition        
            uz_split_z(u_source_pos_index) = source.uz(u_source_sig_index, t_index);
            
        else
            
            % add the source values to the existing field values 
            uz_split_z(u_source_pos_index) = uz_split_z(u_source_pos_index) + source.uz(u_source_sig_index, t_index);
            
        end
    end    
    
    % combine split field components (these variables do not necessarily
    % need to be stored, they could be computed when needed)
    ux_sgx = ux_split_x + ux_split_y + ux_split_z;
    uy_sgy = uy_split_x + uy_split_y + uy_split_z;
    uz_sgz = uz_split_x + uz_split_y + uz_split_z;
    
    % calculate the velocity gradients (these variables do not necessarily
    % need to be stored, they could be computed when needed)
    duxdx = real( ifft( bsxfun(@times, ddx_k_shift_neg, fft(ux_sgx, [], 1)), [], 1));      
    duxdy = real( ifft( bsxfun(@times, ddy_k_shift_pos, fft(ux_sgx, [], 2)), [], 2));
    duxdz = real( ifft( bsxfun(@times, ddz_k_shift_pos, fft(ux_sgx, [], 3)), [], 3));
    
    duydx = real( ifft( bsxfun(@times, ddx_k_shift_pos, fft(uy_sgy, [], 1)), [], 1));
    duydy = real( ifft( bsxfun(@times, ddy_k_shift_neg, fft(uy_sgy, [], 2)), [], 2));
    duydz = real( ifft( bsxfun(@times, ddz_k_shift_pos, fft(uy_sgy, [], 3)), [], 3)); 
    
    duzdx = real( ifft( bsxfun(@times, ddx_k_shift_pos, fft(uz_sgz, [], 1)), [], 1));
    duzdy = real( ifft( bsxfun(@times, ddy_k_shift_pos, fft(uz_sgz, [], 2)), [], 2));
    duzdz = real( ifft( bsxfun(@times, ddz_k_shift_neg, fft(uz_sgz, [], 3)), [], 3));
    
    % update the normal components and shear components of stress tensor
    % using a split field pml
    if flags.kelvin_voigt_model
        
        % compute additional gradient terms needed for the Kelvin-Voigt
        % model
        dduxdxdt = real(ifft( bsxfun(@times, ddx_k_shift_neg, fft( (dsxxdx + dsxydy + dsxzdz) .* rho0_sgx_inv , [], 1 )), [], 1));   
        dduydydt = real(ifft( bsxfun(@times, ddy_k_shift_neg, fft( (dsxydx + dsyydy + dsyzdz) .* rho0_sgy_inv , [], 2 )), [], 2));
        dduzdzdt = real(ifft( bsxfun(@times, ddz_k_shift_neg, fft( (dsxzdx + dsyzdy + dszzdz) .* rho0_sgz_inv , [], 3 )), [], 3));
        
        dduxdydt = real(ifft( bsxfun(@times, ddy_k_shift_pos, fft( (dsxxdx + dsxydy + dsxzdz) .* rho0_sgx_inv , [], 2 )), [], 2));
        dduydxdt = real(ifft( bsxfun(@times, ddx_k_shift_pos, fft( (dsxydx + dsyydy + dsyzdz) .* rho0_sgy_inv , [], 1 )), [], 1)); 
        
        dduxdzdt = real(ifft( bsxfun(@times, ddz_k_shift_pos, fft( (dsxxdx + dsxydy + dsxzdz) .* rho0_sgx_inv , [], 3 )), [], 3));
        dduzdxdt = real(ifft( bsxfun(@times, ddx_k_shift_pos, fft( (dsxzdx + dsyzdy + dszzdz) .* rho0_sgz_inv , [], 1 )), [], 1));
        
        dduydzdt = real(ifft( bsxfun(@times, ddz_k_shift_pos, fft( (dsxydx + dsyydy + dsyzdz) .* rho0_sgy_inv , [], 3 )), [], 3));
        dduzdydt = real(ifft( bsxfun(@times, ddy_k_shift_pos, fft( (dsxzdx + dsyzdy + dszzdz) .* rho0_sgz_inv , [], 2 )), [], 2));
        
        % update the normal shear components of the stress tensor using a
        % Kelvin-Voigt model with a split-field multi-axial pml                 
        sxx_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, ...
                      bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, sxx_split_x))) ...
                      + dt .* (2 .* mu + lambda) .* duxdx ...
                      + dt .* (2 .* eta + chi) .* dduxdxdt)));
        sxx_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, ...
                      bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, sxx_split_y))) ...
                      + dt .* lambda .* duydy ...
                      + dt .* chi .* dduydydt)));
        sxx_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, ...
                      bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, sxx_split_z))) ...
                      + dt .* lambda .* duzdz ...
                      + dt .* chi .* dduzdzdt)));                

        syy_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, ...
                      bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, syy_split_x))) ...
                      + dt .* lambda .* duxdx ...
                      + dt .* chi .* dduxdxdt)));
        syy_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, ...
                      bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, syy_split_y))) ...
                      + dt .* (2 .* mu + lambda) .* duydy ...
                      + dt .* (2 .* eta + chi) .* dduydydt)));
        syy_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, ...
                      bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, syy_split_z))) ...
                      + dt .* lambda .* duzdz ...
                      + dt .* chi .* dduzdzdt)));                  
                  
        szz_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, ...
                      bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, szz_split_x))) ...
                      + dt .* lambda .* duxdx...
                      + dt .* chi .* dduxdxdt)));
        szz_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, ...
                      bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, szz_split_y))) ...
                      + dt .* lambda .* duydy ...
                      + dt .* chi .* dduydydt)));
        szz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, ...
                      bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, szz_split_z))) ...
                      + dt .* (2 .* mu + lambda) .* duzdz ...
                      + dt .* (2 .* eta + chi) .* dduzdzdt)));
                  
        sxy_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x_sgx, ...
                      bsxfun(@times, mpml_z, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x_sgx, sxy_split_x))) ...
                      + dt .* mu_sgxy .* duydx ...
                      + dt .* eta_sgxy .* dduydxdt)));
        sxy_split_y = bsxfun(@times, mpml_z, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_y_sgy, ...
                      bsxfun(@times, mpml_z, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_y_sgy, sxy_split_y))) ...
                      + dt .* mu_sgxy .* duxdy ...
                      + dt .* eta_sgxy .* dduxdydt)));
                  
        sxz_split_x = bsxfun(@times, mpml_y, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_x_sgx, ...
                      bsxfun(@times, mpml_y, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_x_sgx, sxz_split_x))) ...
                      + dt .* mu_sgxz .* duzdx ...
                      + dt .* eta_sgxz .* dduzdxdt)));
        sxz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z_sgz, ...
                      bsxfun(@times, mpml_y, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z_sgz, sxz_split_z))) ...
                      + dt .* mu_sgxz .* duxdz ...
                      + dt .* eta_sgxz .* dduxdzdt)));

        syz_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y_sgy, ...
                      bsxfun(@times, mpml_x, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y_sgy, syz_split_y))) ...
                      + dt .* mu_sgyz .* duzdy ...
                      + dt .* eta_sgyz .* dduzdydt)));
        syz_split_z = bsxfun(@times, mpml_x, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_z_sgz, ...
                      bsxfun(@times, mpml_x, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_z_sgz, syz_split_z))) ...
                      + dt .* mu_sgyz .* duydz ...
                      + dt .* eta_sgyz .* dduydzdt)));                  
        
    else

        % update the normal and shear components of the stress tensor using
        % a lossless elastic model with a split-field multi-axial pml
        sxx_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, ...
                      bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, sxx_split_x))) ...
                      + dt .* (2 .* mu + lambda) .* duxdx )));
        sxx_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, ...
                      bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, sxx_split_y))) ...
                      + dt .* lambda .* duydy )));
        sxx_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, ...
                      bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, sxx_split_z))) ...
                      + dt .* lambda .* duzdz )));

        syy_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, ...
                      bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, syy_split_x))) ...
                      + dt .* lambda .* duxdx)));
        syy_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, ...
                      bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, syy_split_y))) ...
                      + dt .* (2 .* mu + lambda) .* duydy )));
        syy_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, ...
                      bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, syy_split_z))) ...
                      + dt .* lambda .* duzdz )));

        szz_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, ...
                      bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, szz_split_x))) ...
                      + dt .* lambda .* duxdx)));
        szz_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, ...
                      bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, szz_split_y))) ...
                      + dt .* lambda .* duydy )));
        szz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, ...
                      bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, szz_split_z))) ...
                      + dt .* (2 .* mu + lambda) .* duzdz )));

        sxy_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x_sgx, ...
                      bsxfun(@times, mpml_z, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x_sgx, sxy_split_x))) ...
                      + dt .* mu_sgxy .* duydx)));
        sxy_split_y = bsxfun(@times, mpml_z, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_y_sgy, ...
                      bsxfun(@times, mpml_z, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_y_sgy, sxy_split_y))) ...
                      + dt .* mu_sgxy .* duxdy)));

        sxz_split_x = bsxfun(@times, mpml_y, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_x_sgx, ...
                      bsxfun(@times, mpml_y, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_x_sgx, sxz_split_x))) ...
                      + dt .* mu_sgxz .* duzdx)));
        sxz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z_sgz, ...
                      bsxfun(@times, mpml_y, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z_sgz, sxz_split_z))) ...
                      + dt .* mu_sgxz .* duxdz)));

        syz_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y_sgy, ...
                      bsxfun(@times, mpml_x, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y_sgy, syz_split_y))) ...
                      + dt .* mu_sgyz .* duzdy)));
        syz_split_z = bsxfun(@times, mpml_x, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_z_sgz, ...
                      bsxfun(@times, mpml_x, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_z_sgz, syz_split_z))) ...
                      + dt .* mu_sgyz .* duydz)));
                  
    end
              
    % add in the pre-scaled stress source terms
    if flags.source_sxx >= t_index
        if strcmp(source.s_mode, 'dirichlet')
            
            % enforce the source values as a dirichlet boundary condition
            sxx_split_x(s_source_pos_index) = source.sxx(s_source_sig_index, t_index);
            sxx_split_y(s_source_pos_index) = source.sxx(s_source_sig_index, t_index);
            sxx_split_z(s_source_pos_index) = source.sxx(s_source_sig_index, t_index);
            
        else
            
            % add the source values to the existing field values 
            sxx_split_x(s_source_pos_index) = sxx_split_x(s_source_pos_index) + source.sxx(s_source_sig_index, t_index);
            sxx_split_y(s_source_pos_index) = sxx_split_y(s_source_pos_index) + source.sxx(s_source_sig_index, t_index);
            sxx_split_z(s_source_pos_index) = sxx_split_z(s_source_pos_index) + source.sxx(s_source_sig_index, t_index);
            
        end
    end
    if flags.source_syy >= t_index
        if strcmp(source.s_mode, 'dirichlet')
            
            % enforce the source values as a dirichlet boundary condition
            syy_split_x(s_source_pos_index) = source.syy(s_source_sig_index, t_index);
            syy_split_y(s_source_pos_index) = source.syy(s_source_sig_index, t_index);
            syy_split_z(s_source_pos_index) = source.syy(s_source_sig_index, t_index);
            
        else
            
            % add the source values to the existing field values 
            syy_split_x(s_source_pos_index) = syy_split_x(s_source_pos_index) + source.syy(s_source_sig_index, t_index);
            syy_split_y(s_source_pos_index) = syy_split_y(s_source_pos_index) + source.syy(s_source_sig_index, t_index);
            syy_split_z(s_source_pos_index) = syy_split_z(s_source_pos_index) + source.syy(s_source_sig_index, t_index);
            
        end
    end
    if flags.source_szz >= t_index
        if strcmp(source.s_mode, 'dirichlet')
            
            % enforce the source values as a dirichlet boundary condition
            szz_split_x(s_source_pos_index) = source.szz(s_source_sig_index, t_index);
            szz_split_y(s_source_pos_index) = source.szz(s_source_sig_index, t_index);
            szz_split_z(s_source_pos_index) = source.szz(s_source_sig_index, t_index);
            
        else
            
            % add the source values to the existing field values 
            szz_split_x(s_source_pos_index) = szz_split_x(s_source_pos_index) + source.szz(s_source_sig_index, t_index);
            szz_split_y(s_source_pos_index) = szz_split_y(s_source_pos_index) + source.szz(s_source_sig_index, t_index);
            szz_split_z(s_source_pos_index) = szz_split_z(s_source_pos_index) + source.szz(s_source_sig_index, t_index);
            
        end
    end    
    if flags.source_sxy >= t_index
        if strcmp(source.s_mode, 'dirichlet')
            
            % enforce the source values as a dirichlet boundary condition
            sxy_split_x(s_source_pos_index) = source.sxy(s_source_sig_index, t_index);
            sxy_split_y(s_source_pos_index) = source.sxy(s_source_sig_index, t_index);
            
        else
            
            % add the source values to the existing field values 
            sxy_split_x(s_source_pos_index) = sxy_split_x(s_source_pos_index) + source.sxy(s_source_sig_index, t_index);
            sxy_split_y(s_source_pos_index) = sxy_split_y(s_source_pos_index) + source.sxy(s_source_sig_index, t_index);
            
        end
    end
    if flags.source_sxz >= t_index
        if strcmp(source.s_mode, 'dirichlet')
            
            % enforce the source values as a dirichlet boundary condition
            sxz_split_x(s_source_pos_index) = source.sxz(s_source_sig_index, t_index);
            sxz_split_z(s_source_pos_index) = source.sxz(s_source_sig_index, t_index);
            
        else
            
            % add the source values to the existing field values 
            sxz_split_x(s_source_pos_index) = sxz_split_x(s_source_pos_index) + source.sxz(s_source_sig_index, t_index);
            sxz_split_z(s_source_pos_index) = sxz_split_z(s_source_pos_index) + source.sxz(s_source_sig_index, t_index);
            
        end
    end  
    if flags.source_syz >= t_index
        if strcmp(source.s_mode, 'dirichlet')
            
            % enforce the source values as a dirichlet boundary condition
            syz_split_y(s_source_pos_index) = source.syz(s_source_sig_index, t_index);
            syz_split_z(s_source_pos_index) = source.syz(s_source_sig_index, t_index);
            
        else
            
            % add the source values to the existing field values 
            syz_split_y(s_source_pos_index) = syz_split_y(s_source_pos_index) + source.syz(s_source_sig_index, t_index);
            syz_split_z(s_source_pos_index) = syz_split_z(s_source_pos_index) + source.syz(s_source_sig_index, t_index);
            
        end
    end     
    
    % compute pressure from normal components of the stress
    p = -(sxx_split_x + sxx_split_y + sxx_split_z + syy_split_x + syy_split_y + syy_split_z + szz_split_x + szz_split_y + szz_split_z) / 3;
    
    % extract required sensor data from the pressure and particle velocity
    % fields if the number of time steps elapsed is greater than
    % sensor.record_start_index (defaults to 1) 
    if flags.use_sensor && ~flags.elastic_time_rev && (t_index >= sensor.record_start_index)
    
        % update index for data storage
        file_index = t_index - sensor.record_start_index + 1;
        
        % store the acoustic pressure if using a transducer object
        if flags.transducer_sensor
            error('Using a kWaveTransducer for output is not currently supported.');
        end
        
        % run sub-function to extract the required data
        sensor_data = kspaceFirstOrder_extractSensorData(3, sensor_data, file_index, sensor_mask_index, flags, record, p, ux_sgx, uy_sgy, uz_sgz);
        
        % check stream to disk option
        if flags.stream_to_disk
            error('''StreamToDisk'' input is not currently supported.');
        end
        
    end
    
    % estimate the time to run the simulation
    if t_index == ESTIMATE_SIM_TIME_STEPS
                
        % display estimated simulation time
        disp(['  estimated simulation time ' scaleTime(etime(clock, loop_start_time)*index_end/t_index) '...']);

        % check memory usage
        kspaceFirstOrder_checkMemoryUsage; 
        
    end    
    
    % plot data if required
    if flags.plot_sim && (rem(t_index, plot_freq) == 0 || t_index == 1 || t_index == index_end) 

        % update progress bar
        waitbar(t_index/kgrid.Nt, pbar);
        drawnow;   

        % ensure p is cast as a CPU variable and remove the PML from the
        % plot if required
        if strcmp(data_cast, 'gpuArray')
            sii_plot = double(gather(p(x1:x2, y1:y2, z1:z2)));
            sij_plot = double(gather((...
            sxy_split_x(x1:x2, y1:y2, z1:z2) + sxy_split_y(x1:x2, y1:y2, z1:z2) + ...
            sxz_split_x(x1:x2, y1:y2, z1:z2) + sxz_split_z(x1:x2, y1:y2, z1:z2) + ...
            syz_split_y(x1:x2, y1:y2, z1:z2) + syz_split_z(x1:x2, y1:y2, z1:z2) )/3));
        else
            sii_plot = double(p(x1:x2, y1:y2, z1:z2));  
            sij_plot = double((...
            sxy_split_x(x1:x2, y1:y2, z1:z2) + sxy_split_y(x1:x2, y1:y2, z1:z2) + ...
            sxz_split_x(x1:x2, y1:y2, z1:z2) + sxz_split_z(x1:x2, y1:y2, z1:z2) + ...
            syz_split_y(x1:x2, y1:y2, z1:z2) + syz_split_z(x1:x2, y1:y2, z1:z2) )/3);
        end

        % update plot scale if set to automatic or log
        if flags.plot_scale_auto || flags.plot_scale_log
            kspaceFirstOrder_adjustPlotScale;
        end  

        % add display mask onto plot
        if strcmp(display_mask, 'default')
            sii_plot(sensor.mask(x1:x2, y1:y2, z1:z2) ~= 0) = plot_scale(2);
            sij_plot(sensor.mask(x1:x2, y1:y2, z1:z2) ~= 0) = plot_scale(end);
        elseif ~strcmp(display_mask, 'off')
            sii_plot(display_mask(x1:x2, y1:y2, z1:z2) ~= 0) = plot_scale(2);
            sij_plot(display_mask(x1:x2, y1:y2, z1:z2) ~= 0) = plot_scale(end);
        end
        
        % update plot
        planeplot(scale .* kgrid.x_vec(x1:x2), scale .* kgrid.y_vec(y1:y2), scale .* kgrid.z_vec(z1:z2), ...
            sii_plot, sij_plot, '', plot_scale, prefix, COLOR_MAP);

        % save movie frames if required
        if flags.record_movie
            
            % set background color to white
            set(gcf, 'Color', [1 1 1]);

            % save the movie frame
            writeVideo(video_obj, getframe(gcf));
            
        end        
        
        % update variable used for timing variable to exclude the first
        % time step if plotting is enabled
        if t_index == 1
            loop_start_time = clock;
        end
        
    end
end

% update command line status
disp(['  simulation completed in ' scaleTime(toc)]);

% =========================================================================
% CLEAN UP
% =========================================================================

% clean up used figures
if flags.plot_sim
    close(img);
    close(pbar);
    drawnow;
end

% save the movie frames to disk
if flags.record_movie
    close(video_obj);
end

% save the final acoustic pressure if required
if flags.record_p_final || flags.elastic_time_rev
    sensor_data.p_final = p(record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside, record.z1_inside:record.z2_inside);
end

% save the final particle velocity if required
if flags.record_u_final
    sensor_data.ux_final = ux_sgx(record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside, record.z1_inside:record.z2_inside);
    sensor_data.uy_final = uy_sgy(record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside, record.z1_inside:record.z2_inside);
    sensor_data.uz_final = uz_sgz(record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside, record.z1_inside:record.z2_inside);
end

% run subscript to cast variables back to double precision if required
if flags.data_recast
    kspaceFirstOrder_dataRecast;
end

% run subscript to compute and save intensity values
if flags.use_sensor && ~flags.elastic_time_rev && (flags.record_I || flags.record_I_avg)
    save_intensity_matlab_code = true;
    kspaceFirstOrder_saveIntensity;
end

% reorder the sensor points if a binary sensor mask was used for Cartesian
% sensor mask nearest neighbour interpolation (this is performed after
% recasting as the GPU toolboxes do not all support this subscript)
if flags.use_sensor && flags.reorder_data
    kspaceFirstOrder_reorderCartData;
end

% filter the recorded time domain pressure signals if transducer filter
% parameters are given 
if flags.use_sensor && ~flags.elastic_time_rev && isfield(sensor, 'frequency_response')
    sensor_data.p = gaussianFilter(sensor_data.p, 1/kgrid.dt, sensor.frequency_response(1), sensor.frequency_response(2));
end

% reorder the sensor points if cuboid corners is used (outputs are indexed
% as [X, Y, Z, T] or [X, Y, Z] rather than [sensor_index, time_index]
if flags.cuboid_corners
    kspaceFirstOrder_reorderCuboidCorners;
end

if flags.elastic_time_rev
    
    % if computing time reversal, reassign sensor_data.p_final to
    % sensor_data
    sensor_data = sensor_data.p_final;
    
elseif ~flags.use_sensor
    
    % if sensor is not used, return empty sensor data
    sensor_data = [];
    
elseif ~isfield(sensor, 'record') && ~flags.cuboid_corners
    
    % if sensor.record is not given by the user, reassign sensor_data.p to
    % sensor_data
    sensor_data = sensor_data.p;
    
end

% update command line status
disp(['  total computation time ' scaleTime(etime(clock, start_time))]);

% switch off log
if flags.create_log
    diary off;
end

function planeplot(x_vec, y_vec, z_vec, s_normal_plot, s_shear_plot, data_title, plot_scale, prefix, color_map)
% Subfunction to produce a plot of the elastic wavefield
   
% plot normal stress
subplot(2, 3, 1);
imagesc(y_vec, x_vec, squeeze(s_normal_plot(:, :, round(end/2))), plot_scale(1:2));
title('Normal Stress (x-y plane)'), axis image;

subplot(2, 3, 2);
imagesc(z_vec, x_vec, squeeze(s_normal_plot(:, round(end/2), :)), plot_scale(1:2));
title('Normal Stress  (x-z plane)'), axis image;

subplot(2, 3, 3);
imagesc(z_vec, y_vec, squeeze(s_normal_plot(round(end/2), :, :)), plot_scale(1:2));
title('Normal Stress  (y-z plane)'), axis image;

% plot shear stress
subplot(2, 3, 4);
imagesc(y_vec, x_vec, squeeze(s_shear_plot(:, :, round(end/2))), plot_scale(end - 1:end));
title('Shear Stress (x-y plane)'), axis image;

subplot(2, 3, 5);
imagesc(z_vec, x_vec, squeeze(s_shear_plot(:, round(end/2), :)), plot_scale(end - 1:end));
title('Shear Stress (x-z plane)'), axis image;

subplot(2, 3, 6);
imagesc(z_vec, y_vec, squeeze(s_shear_plot(round(end/2), :, :)), plot_scale(end - 1:end));
title('Shear Stress (y-z plane)'), axis image;

xlabel(['(All axes in ' prefix 'm)']);
colormap(color_map); 
drawnow;