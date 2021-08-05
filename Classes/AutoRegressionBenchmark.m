classdef AutoRegressionBenchmark
    %AutoRegressionBenchmark A Class for testing Algorithms or ensambles of
    %Algorithms for autoregessive timeseries forecasting
    %   Requierments:
    %   + Data must contain a Variable named Date of type DateTime
    %   + Autoregressive Targets names must begin with 'Next_'
    %
    
    properties (SetAccess='public')
        BenchmarkSettings struct %A Struct altering properties such as skipping timesteps
        Data table %Table containing all Features and Targets and a DateTime Variable named Date
        ErrorLog table %A Tabel Containing all Results of the last benchmark run
        Ensamble AutoRegressor
    end
    
     properties (SetAccess='private')
        delayedInput double%for using previous timesteps as input %currently not supportet
    end
    
    methods (Hidden=0, Access='public')
        function obj = AutoRegressionBenchmark(Data,AutoRegressor,BenchmarkSettings)
            %AutoRegressionBenchmark Benchmark Autoregressive Algorithms
            %   Detailed explanation goes here
            
            obj.BenchmarkSettings=BenchmarkSettings;
            obj.Data=Data;
            obj.Ensamble=AutoRegressor;
%            obj=obj.verifyDataset(); %Checks for errors in the Dataset
            
            %for using previous timesteps as input %currently not supportet
            obj.delayedInput=0;
            
        end
        function obj = benchmark(obj)
            %benchmark Trains Models to predict the values of the following
            %   Timestep. Those predictions will be used to predict the as
            %   input argument for the following prediction
            
            %Verifiy Dataset
%            obj.verifyDataset();
            
            % Determine which timesteps to test
            timesteps_to_test=[];
            IndicesOfUniterrupedSequences=obj.findUninterruptedSequences();
            
            for seq=1:size(IndicesOfUniterrupedSequences,1)
                
                start_of_seq=IndicesOfUniterrupedSequences(seq,1)+max(obj.BenchmarkSettings.InitialTrainingSamples,obj.delayedInput);
                end_of_seq=(IndicesOfUniterrupedSequences(seq,2)-obj.BenchmarkSettings.ClosedLoopTimeHorizion)+1;
                stepsize=obj.BenchmarkSettings.Timesteps_skipped+1;
                
                if start_of_seq<end_of_seq
                    new_timesteps=start_of_seq:stepsize:end_of_seq;
                    timesteps_to_test=[timesteps_to_test,new_timesteps];
                end
            end
            
            last_trained=-Inf;
            %for console output
            output_dialog_flag=0;
            %clear previes Results
            obj.ErrorLog=[];
            
            %test each timestep
            for i=1:length(timesteps_to_test)
                
                t=timesteps_to_test(i);
                
                %Prepare table for prediction
                [input_tbl, ground_truth]=obj.prepareTableForPrediction(t);
                
                % Train all Models
                if t-last_trained>=obj.BenchmarkSettings.RetrainFrequency
                obj.Ensamble=obj.Ensamble.train(obj.Data(1:t-1,:));
                last_trained=t;
                end
                %Make Predictions
                output_tbl=obj.Ensamble.predict(input_tbl);
                predictions=output_tbl(end-(size(ground_truth,1)-1):end,obj.trimNext_(obj.Ensamble.All_targets));
                %Store Errors and Performance Metrics
                UserData=cellfun(@(x)x.UserData,obj.Ensamble.Models,'UniformOutput',false);
                
                new_row=obj.testLog(t,predictions, ground_truth,UserData);
                
                if isempty(obj.ErrorLog)
                    obj.ErrorLog=new_row;
                else
                    obj.ErrorLog=[obj.ErrorLog;new_row];
                end
                if obj.BenchmarkSettings.verbose && ...
                        (i./length(timesteps_to_test)*100)>=output_dialog_flag
                    fprintf("Progress %i%% \n",output_dialog_flag)
                    output_dialog_flag=output_dialog_flag+10;
                end
            end
            
            
            
            
            
            obj.ErrorLog=rmmissing(obj.ErrorLog);
        end
        function dispResults(obj)
            %dispResults Some plots and statistics from the last benchmark
            if isempty(obj.ErrorLog)
                warning("No Results to display")
            else
                figure
                h=plot(cell2mat(obj.ErrorLog.RMSE));
                
                hold on
                means=mean(cell2mat(obj.ErrorLog.RMSE));
                arrayfun(@(x)yline(x,'--'),means);
                hold off
                ylabel("RMSE")
                xlabel("Timestep")
                legend(h,obj.ErrorLog.Error(1,:).Properties.VariableNames,'Interpreter',"none")
                
                figure
                
                E=cell2mat(cellfun(@(x)mean(x,1),obj.ErrorLog.ErrorRelative,'UniformOutput',false));
                
                h=plot(E);
                
                hold on
                means=mean(cell2mat(obj.ErrorLog.RMSE));
                arrayfun(@(x)yline(x,'--'),means);
                hold off
                ylabel("Relative Error")
                xlabel("Timestep")
                legend(h,obj.ErrorLog.Error(1,:).Properties.VariableNames,'Interpreter',"none")
                
            end
        end
    end
    
    methods (Hidden=1, Access = 'private' )
        function obj = verifyDataset(obj)
            %verifyDataset checks the dataset for errors
            %   Checks for irregular timing, missing values
            
            re=1;
            
            if sum(obj.Data.Properties.VariableNames=="Date")~=1
                error("Data must contain a Varaible with the name 'Date'")
            end
            
            if length(unique(diff(obj.Data.Date)))~=1
                warning("Timestep Size is irregular")
                re=0;
            end
            
            temp=cellfun(@(x) x.Features,obj.Ensamble.Models,'UniformOutput',false);
            All_features=unique([temp{:}]);
            if sum(ismissing(obj.Data(:,unique(All_features))),'all')~=0
                warning("Some Features have Missing entires")
            end
            
            charArr=char(obj.Ensamble.All_targets);
            for j=1:size(charArr,3)
                if (isequal(charArr(1,1:5,j),'Next_'))
                    charArr(1,1:5,j)=' ';
                    Target_variables(j)=strtrim(string(charArr(1,:,j)));
                end
            end
            
            if sum(ismissing(obj.Data(:,Target_variables)),'all')~=0
                warning("Some Targets have Missing entires")
                re=0;
            end
            
        end
        
        function [tbl, ground_truth] = prepareTableForPrediction(obj,timestep)
            %prepareTableForPrediction Summary of this method goes here
            %   Detailed explanation goes here
            
            %get relevant section of the table
            startIndex = timestep-(obj.delayedInput+1);
            EndIndex = timestep+(obj.BenchmarkSettings.ClosedLoopTimeHorizion-1);
            tbl=obj.Data(startIndex:EndIndex,:);
            ground_truth=tbl(obj.delayedInput+2:end,obj.trimNext_(obj.Ensamble.All_targets));
            %fill targets with nans
            for i=1:numel(obj.Ensamble.All_targets)
                tbl(obj.delayedInput+2:end,obj.trimNext_(obj.Ensamble.All_targets(i)))=...
                    array2table(NaN([obj.BenchmarkSettings.ClosedLoopTimeHorizion,1]));
            end
        end

        function intervals = findUninterruptedSequences(obj)
            %findUninterruptedSequences Returns an n by 2 Array with n
            %steady time Sequences. The first colum contains the starting
            %incices of each sequence and the second colum the last incices
            %of the interval
            
            dt=datenum(diff(obj.Data.Date));
            
            %Determine most common time step size
            timestep=mode(dt);
            
            regular=false;
            starts=[];
            ends=[];
            
            for i=1:length(dt)
                %ends a sequence
                if dt(i)~=timestep && regular
                    regular = false;
                    ends=[ends;i-1];
                end
                %starts a sequence
                if dt(i)==timestep && ~regular
                    regular = true;
                    starts=[starts;i];
                end
            end
            %add ending index if necessary
            if regular && dt(i)==timestep
                ends=[ends;length(obj.Data.Date)-1];
            end
            intervals=[starts, ends];
            
            
        end
        function [LOG] = testLog(obj,Timestep,Predictions,GroundTruth,UserData)
            %testLog Summary of this method goes here
            %   Detailed explanation goes here
            
            Date=obj.Data.Date(Timestep);
            % TODO CHECKEN OB DIE REIHENFOLGE DER TARGETS GLEICH IST!!!
            GroundTruth=table2array(GroundTruth);
            Predictions=table2array(Predictions);
            Error=(GroundTruth-Predictions);
            ErrorRelative=Error./max(GroundTruth,abs(Error));
            ErrorDynamic=Error./(max(GroundTruth,[],1)-min(GroundTruth,[],1));
            RMSE=sqrt(mean(Error.^2,1));
            MAPE=100/obj.BenchmarkSettings.ClosedLoopTimeHorizion * sum(abs(Error)./GroundTruth,1);
            dtw3=dtw(Predictions,GroundTruth,3);
            
            sz=size(Error);
            
            Error=cell2table(mat2cell(Error,sz(1),ones(1,sz(2))),'VariableNames',cellstr(""+obj.Ensamble.All_targets));
            
            if isempty(UserData)
                UserData={[]};
            end
            
            
            LOG=table(Timestep,Date,{GroundTruth},{Predictions},Error,{ErrorRelative},{ErrorDynamic},{RMSE},{dtw3},UserData);
            LOG.Properties.VariableNames{3} = 'GroundTruth';
            LOG.Properties.VariableNames{4} = 'Predictions';
            LOG.Properties.VariableNames{5} = 'Error';
            LOG.Properties.VariableNames{6} = 'ErrorRelative';
            LOG.Properties.VariableNames{7} = 'ErrorDynamic';
            LOG.Properties.VariableNames{8} = 'RMSE';
            LOG.Properties.VariableNames{9} = 'dtw3';
            LOG.Properties.VariableNames{10} = 'UserData';
        end
        
        function names=trimNext_(obj,Array)
            charArr=char(Array);
            for j=1:size(charArr,3)
                if (isequal(charArr(1,1:5,j),'Next_'))
                    charArr(1,1:5,j)=' ';
                    names(j)=strtrim(string(charArr(1,:,j)));
                end
            end
        end
        
    end
end

