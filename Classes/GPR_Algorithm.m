classdef GPR_Algorithm < ForecastingAlgorithmInterface
    %GPR_Algorithm General Single Target Gaussian Process for timeseries
    
    properties
        GPRmodel
        DataSelection
        TrainOptions struct
        
    end
    
    methods
        function obj = GPR_Algorithm(Features,Targets,...
                DataSelection,TrainOptions)
            %GPR_Algorithm Construct an instance of this class
            
            obj.Features = Features;
            obj.Targets = Targets;

            if isempty(DataSelection)
                DataSelection=@(tbl)tbl;
            end
            obj.DataSelection = DataSelection;
            obj.TrainOptions = TrainOptions;
        end
        
        
        function tbl = predict(obj,tbl)
            %predict Summary of this method goes here
            %   Detailed explanation goes here
            
            %in reihenfolge der obj.targets
            prediction(:,1) = predict(obj.GPRmodel,tbl);
            
            for i=1:length(obj.Targets)
                tbl(:,obj.Targets(i))=table(prediction(:,i));
            end
            
        end
        
        function tbl = predictCL(obj,tbl)
            %predictCL Summary of this method goes here
            %   Detailed explanation goes here
            
            [rows,~] = find(ismissing(tbl(:,obj.Targets)));
            rows=unique(rows)';
            
            for i=rows
                tbl(i,:)=obj.predict(tbl(i,:));
                if i<height(tbl)
                    charArr=char(obj.Targets);
                    if (isequal(charArr(1:5),'Next_'))
                        charArr(1:5)=[];
                        temp=string(charArr);    
                        tbl(i+1,temp)=tbl(i,obj.Targets);
                    end
                    
                end
            end
            
        end
        
        function obj=train(obj,tbl)
            %train a GPR to predict the target
            %   Detailed explanation goes here
            
            %Select Trainingdata
            tbl=obj.DataSelection(tbl);
            
            %Specifiy Target and Features
            formula=obj.Targets+"~";
            for i=1:length(obj.Features)
                formula=formula+obj.Features(i);
                if i < length(obj.Features)
                    formula=formula+"+";
                end
            end
            
            %Prepare Name Value Pairs
            if isempty(obj.TrainOptions)
                obj.GPRmodel= fitrgp(tbl,formula);
            else
                nvPairs = reshape([fieldnames(obj.TrainOptions),...
                struct2cell(obj.TrainOptions)]',1,[]);
            
            %Train Model
            obj.GPRmodel= fitrgp(tbl,formula,nvPairs{:});
            end

            %SaveModelParameters
            obj.UserData={obj.GPRmodel.ModelParameters};
        end
        
    end
end

