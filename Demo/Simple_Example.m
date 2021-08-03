addpath("Classes\","Classes\Models\","Data\")
load("energydatacomplete.mat")
data_training=energydatacomplete(1:400,:); %One Day
data_test=energydatacomplete(400:500,:); %Following hour
% Dataset Discription 
disp(head(energydatacomplete,4))


%create an basic Gaussian Process for prediction of humidity

Ensamble=AutoRegressor()

%Define and add a Model to the Ensamble
Features=["T_out","T2","T6"];
Targets="Next_T6";
TrainOpt.Kernelfunction='ARDExponential';
simpleModel=GP(Features,Targets,TrainOpt);
Ensamble = Ensamble.addModel(simpleModel);
%Train and Make a Predition
Ensamble=Ensamble.train(data_training);

%%
% Prepare a Table for Prediction by setting targets to NaN or empty values
input_table=data_test;
predict_rows=2:height(data_test);
input_table(predict_rows,"T6")=array2table(nan([length(predict_rows) 1]));
disp(head(input_table(:,["Date",simpleModel.Features])))
% By predicting these Values will be filled out
predicted_table=Ensamble.predict(input_table);
disp(head(predicted_table(:,["Date",simpleModel.Features])))
%Benchmark your system
testing_options=struct(...
    'InitialTrainingSamples',6,...
    'ClosedLoopTimeHorizion',6,...
    'UseParallel',0,...
    'verbose',1,...
    'Timesteps_skipped',1);
Bench= AutoRegressionBenchmark(data_training,Ensamble,testing_options);
Bench=Bench.benchmark;
Bench.dispResults
%Add  more Models

%benchmark your System