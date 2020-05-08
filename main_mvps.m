%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The implementation of the paper:                               %
% "Inferring Super-Resolution Depth from a Moving Light-Source   %
% Enhanced RGB-D Sensor: a Variational Approach"                 %    
% Lu Sang, Bjoern Haefner, Daniel Cremers                        %
%                                                                %
% The code can only be used for research purposes.               %
%                                                                %
% Computer Vision Group, TUM                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
clc

addpath(strcat(pwd,'/data/'))
addpath(genpath(strcat(pwd,'/DVO/build')));
addpath(genpath(strcat(pwd,'/src')));

%% main part of the demo code

% the algorithm needs create 2 new folders under current folder path so
% please make sure the writting permission.

% we provide one synthetic data 'joyful_yell' and one real world data 'hat' for testing
% for the 'joyful_yell' you can choose any image as the reference frame
% (refFrame) since there is mask for every image. For the 'hat' currently
% we only generate 2 masks for image 3 and 12, so please choose one of this
% as refFrame.


dataset = 'hat'; % 'joyful_yell' or 'hat'
refFrame = 3;
SF = 2;

% input dataset check
if ~strcmp(dataset,'hat') && ~strcmp(dataset,'joyful_yell')
    error("Invalid dataset name! please choose between 'joyful_yell' and 'hat'.");
elseif strcmp(dataset,'hat') && ~exist('refFrame','var')
    fprintf("warning: reference frame is not give, use a default one.");
    refFrame = 3;
elseif strcmp(dataset,'hat') && (refFrame~=3 && refFrame~=12)
    fprintf("warning: reference frame is not valid, choose between 3 or 12. For now use a default one.");
    refFrame = 3;
elseif strcmp(dataset,'joyful_yell') && ~exist('refFrame','var')
    fprintf("warning: reference frame is not give, use a default one.");
    refFrame = 1;
elseif strcmp(dataset,'joyful_yell') && ~exist('SF','var')
    SF = 2;
    fprintf("warning: scaling factor is not give, use a default one.");
elseif strcmp(dataset,'joyful_yell') && (SF~=2 && SF~=4)
    SF = 2;
    fprintf("warning: scaling factor is not valid, use 2 or 4, For now set to a default one.");
end

% set parameters 
params = struct;
params.SF = SF;
params.refFrame = refFrame;
if strcmp(dataset,'joyful_yell')
    params.realdata = 0;
else
    params.realdata = 1;
    SF = 2;
end
% parameters which can be tune
% method canbe choosen from ["Cauchy","L2","Welsh","GM","Tukey","Huber"] 
% pay attention to upper case of the letters!

params.tau = 1;
params.max_iter = 30;
params.tol = 5e-3;
params.method = 'Cauchy';
params.method_delta = 0.5;
params.do_display = 1;

[z, rho, N, s, pose_est] = MultiViewSR(pwd, dataset, params);


