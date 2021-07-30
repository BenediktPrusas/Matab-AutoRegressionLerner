classdef AutoRegressionBenchmark
    %AutoRegressionBenchmark A Class for testing Algorithms or ensambles of
    %Algorithms for autoregessive timeseries forecasting
    %   Requierments:
    %   + Data must contain a Variable named Date of type DateTime
    %   + Autoregressive Targets names must begin with 'Next_'
    %
    
    properties (SetAccess='public')
        BenchmarkSettings struct %A Struct altering properties such as skipping timesteps
        Algorithms cell %CellArrays of Algorithms which inherit from Forecasting AlgorithmInterfache
        Data table %Table containing all Features and Targets and a DateTime Variable named Date
        ErrorLog table %A Tabel Containing all Results of the last benchmark run
    end
    properties (SetAccess='private',SetObservable=false)
        All_targets string
        All_features string
    end
    
    methods (Hidden=0, Access='public')
        function obj = AutoRegressionBenchmark(Data,Algorithm,BenchmarkSettings)
            %AutoRegressionBenchmark Benchmark Autoregressive Algorithms
            %   Detailed explanation goes here
            
            obj.BenchmarkSettings=BenchmarkSettings;
            obj.Data=Data;
            obj=obj.verifyDataset(); %Checks for errors in the Dataset
            obj=obj.addAlgorithm(Algorithm);
            
        end
        function obj = benchmark(obj)
            %benchmark Trains Models to predict the values of the following
            %   Timestep. Those predictions will be used to predict the as
            %   input argument for the following prediction
            
            %Verifiy Dataset
            obj.verifyDataset();
            %Update Prediction Order
            PredictionOrder=obj.determinePredictionOrder();
            
            % Determine which timesteps to test
            timesteps_to_test=[];
            IndicesOfUniterrupedSequences=obj.findUninterruptedSequences();
            
            for seq=1:size(IndicesOfUniterrupedSequences,1)
                
                start_of_seq=IndicesOfUniterrupedSequences(seq,1)+obj.BenchmarkSettings.InitialTrainingSamples;
                end_of_seq=(IndicesOfUniterrupedSequences(seq,2)-obj.BenchmarkSettings.ClosedLoopTimeHorizion)+1;
                stepsize=obj.BenchmarkSettings.Timesteps_skipped+1;
                
                if start_of_seq<end_of_seq
                    new_timesteps=start_of_seq:stepsize:end_of_seq;
                    timesteps_to_test=[timesteps_to_test,new_timesteps];
                end
            end
            
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
                for  a=1:length(obj.Algorithms)
                    obj.Algorithms{a}.train(obj.Data(1:t-1,:));
                end
                
                %Make Predictions
                for j=1:obj.BenchmarkSettings.ClosedLoopTimeHorizion
                    for a=PredictionOrder
                        %Predict
                        input_tbl(j,:)=obj.Algorithms{a}.predict(input_tbl(j,:));
                        predictions=input_tbl(:,obj.All_targets);
                    end
                    %fill out the features of the next timestep
                    if j<height(input_tbl)
                        charArr=char(obj.All_targets);
                        for j=1:size(charArr,3)
                            if (isequal(charArr(1,1:5,j),'Next_'))
                                charArr(1,1:5,j)=' ';
                                feature_name=strtrim(string(charArr(1,:,j)));
                                input_tbl(j+1,feature_name)=input_tbl(j,strtrim(charArr(1,:,j)));
                            end
                        end
                    end
                    
                end
                %Store Errors and Performance Metrics
                UserData=cellfun(@(x)x.UserData,obj.Algorithms,'UniformOutput',false);
                
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
        function obj = addAlgorithm(obj,UserIn)
            %addAlgorithm Adds an Algoritm or cell Array of Algorithms to
            %the benchmark object
            s=superclasses(UserIn);
            if ~isempty(s) && s{1}=="ForecastingAlgorithmInterface"
                obj.Algorithms{end+1}=UserIn;
                obj.All_targets(end+1)=UserIn.Targets;
                obj.All_features=union(obj.All_features,UserIn.Features);
                obj.determinePredictionOrder;
            else
                %case when a cell array of algorithms is delivered
                if class(UserIn)=='cell'
                    if length(size(UserIn))==2 && (sum(size(UserIn)==1)>0)
                        for i=1:length(UserIn)
                            s=superclasses(UserIn{i});
                            if ~isempty(s) &&s{1}=="ForecastingAlgorithmInterface"
                                obj.Algorithms{end+1}=UserIn{i};
                                obj.All_targets(end+1)=UserIn{i}.Targets;
                                obj.All_features=union(obj.All_features,UserIn{i}.Features);
                            else
                                error("Algorithm object must have ForecastingAlgorithmInterface as superclass")
                            end
                        end
                        obj.determinePredictionOrder();
                    end
                end
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
            
            temp=cellfun(@(x) x.Features,obj.Algorithms,'UniformOutput',false);
            obj.All_features=unique([temp{:}]);
            if sum(ismissing(obj.Data(:,unique(obj.All_features))),'all')~=0
                warning("Some Features have Missing entires:")
                warning(obj.Algorithms.Features((sum(ismissing(obj.Data(:,obj.Algorithms.Features)))>0)'))
                re=0;
            end
            
            temp=cellfun(@(x) x.Features,obj.Algorithms,'UniformOutput',false);
            obj.All_targets=unique([temp{:}]);
            if sum(ismissing(obj.Data(:,obj.All_targets)),'all')~=0
                warning("Some Targets have Missing entires:")
                warning(obj.All_targets((sum(ismissing(obj.Data(:,obj.All_targets)))>0)'))
                re=0;
            end
            
        end
        
        function [tbl, ground_truth] = prepareTableForPrediction(obj,timestep)
            %prepareTableForPrediction Summary of this method goes here
            %   Detailed explanation goes here
            
            %get relevant section of the table
            startIndex = timestep;
            EndIndex = timestep+obj.BenchmarkSettings.ClosedLoopTimeHorizion-1;
            tbl=obj.Data(startIndex:EndIndex,:);
            ground_truth=tbl(:,obj.All_targets);
            %fill targets with nans
            for i=1:numel(obj.All_targets)
                tbl(:,obj.All_targets(i))=...
                    array2table(NaN([obj.BenchmarkSettings.ClosedLoopTimeHorizion,1]));
            end
        end
        
        function PredictionOrder = determinePredictionOrder(obj)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            PredictionOrder=[];
            
            %contruct graph
            
            if numel(obj.All_targets)<numel(obj.Algorithms)
                error("Some Target gets predicted by more then one Algorithm")
            end
            
            graph=zeros(length(obj.All_targets));
            
            for i=1:length(obj.Algorithms)
                [dependent,~,indices]=intersect(obj.Algorithms{i}.Features,obj.All_targets);
                if isempty(dependent)
                    PredictionOrder(end+1)=i;
                else
                    graph(indices,i)=1;
                end
            end
            %iterate over graph
            if sum(graph,'all') >0
                if obj.BenchmarkSettings.verbose
                    fprintf("Some predictions are based on Targets \n")
                end
                if sum(diag(graph))>0
                    error("A Target prediction depends on it self")
                end
                while true
                    resolved=find(sum(graph,1)==0);
                    if ~isempty(resolved)
                        graph(resolved,:)=0;
                        
                        %Add new resolved to the list
                        new=setdiff(resolved,PredictionOrder);
                        if ~isempty(new)
                            PredictionOrder=[PredictionOrder,new];
                        elseif sum(diag(graph),'all')==0
                            %add remaining to the prediction order
                            PredictionOrder=[PredictionOrder,setdiff(1:length(obj.Algorithms),PredictionOrder)];
                            break;
                        else
                            error("Dependencies cannot be resolved")
                        end
                    end
                end
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
            ErrorRelative=Error./max(GroundTruth,Error);
            ErrorDynamic=Error./(max(GroundTruth,[],1)-min(GroundTruth,[],1));
            RMSE=sqrt(mean(Error.^2,1));
            MAPE=100/obj.BenchmarkSettings.ClosedLoopTimeHorizion * sum(abs(Error)./GroundTruth,1);
            dtw3=dtw(Predictions,GroundTruth,3);
            
            sz=size(Error);
            
            Error=cell2table(mat2cell(Error,sz(1),ones(1,sz(2))),'VariableNames',cellstr(""+obj.All_targets));
            
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
    end
end

