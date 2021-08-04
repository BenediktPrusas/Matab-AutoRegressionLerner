%% Advanced Example
% This example shows basic use of the Classes

addpath("Classes\","Classes\Models\","Data\")
%% Example Dataset
% The Dataset from Luis Candanedo, luismiguel.candanedoibarra '@' umons.ac.be, 
% University of Mons (UMONS). 
% 
% _*Data Set Information:*_
% 
% _The data set is at 10 min for about 4.5 months. The  house temperature and 
% humidity conditions were monitored with a ZigBee  wireless sensor network. Each 
% wireless node transmitted the temperature  and humidity conditions around 3.3 
% min. Then, the wireless data was  averaged for 10 minutes periods. The energy 
% data was logged every 10  minutes with m-bus energy meters. Weather from the 
% nearest airport  weather station (Chievres Airport, Belgium) was downloaded 
% from a public data set from Reliable Prognosis (rp5.ru), and merged together 
% with the experimental data sets using the date and time column. Two random  
% variables have been included in the data set for testing the regression  models 
% and to filter out non predictive attributes (parameters)._
% 
% % 
% Cited from from <https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction# 
% https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction#>
% 
% 
load("energydatacomplete.mat")
%energydatacomplete=timetable2table(retime(table2timetable(energydatacomplete),"hourly"))
energydatacomplete(2001:end,:)=[];
disp(head(energydatacomplete,4))
%% Ensambles of Models and Training Options
% You can mix and match diffrent models for diffrent autoregressive Variables.
% 
% First we create again a linear Model to predict T6

%Create an AutoRegressor Object
Ensamble=AutoRegressor();

%Define and add a Model to the Ensamble
Md_T.Features=["T_out","T"+(1:9)];
Md_T.Targets="Next_T"+(1:9);
Md_T.Model=MLP(Md_T.Features,Md_T.Targets,[9 9]);
Ensamble = Ensamble.addModel(Md_T.Model);
%% 
% We can assume T_out to be known, as it can be derived from the local weather 
% forecast.
% 
% But we base our prediction of T6 on T2. So we need a model to also predict 
% T2 in an auto regressive manner.Lets create an Support Vector Machine for that 
% purpose:


%% 
% You can specifiy Training Options by creating a struct, where the field names 
% and values are valid name value paris for the fitrsvm function.


%% 
% 

%% Benchmarking
% You can benchmark your system, by repeativly makeing autoregressive Predictions 
% an observe the behaviour as in an production enviorment

testing_options=struct(...
    'InitialTrainingSamples',6*24,...
    'ClosedLoopTimeHorizion',24*6,...
    'UseParallel',0,...
    'verbose',1,...
    'Timesteps_skipped',5,...
    'RetrainFrequency',6*24);
Bench= AutoRegressionBenchmark(energydatacomplete,Ensamble,testing_options);
Bench=Bench.benchmark;
%% 
% To quickly visualise the benchmark Results you can use the dispResults function

Bench.dispResults
%% 
% For a more analysis statistics can be derived from the Data in the ErrorLog 
% property of the benchmark object

disp(head(Bench.ErrorLog))