classdef LR < AutoRegressionModel
    %LR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        LRmodel
        TrainOptions struct
    end
    
    methods
        function obj = LR(Features,Targets,TrainOptions,DataSelection)
            %LR Construct an instance of this class
            %   Detailed explanation goes here
            obj.Features = Features;
            
            %default for data selection
            if nargin < 4 || isempty(DataSelection)
                DataSelection=@(tbl)tbl;
            end
            
            obj.DataSelection = DataSelection;
            
            if nargin < 3 || isempty(TrainOptions)
                TrainOptions=struct();
            end
            
            obj.TrainOptions = TrainOptions;
            obj.UserData=[];
        end
        
       function tbl=predict(obj,tbl)

            if isempty(obj.LRmodel)
                error("Model has not been trained before prediction")
            else
                pred=obj.LRmodel.predict(tbl);
                tbl.(obj.Targets)=pred;
            end
            
       end
       
       function obj = train(obj,tbl)
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
                obj.LRmodel= fitlm(tbl,formula);
            else
                nvPairs = reshape([fieldnames(obj.TrainOptions),...
                    struct2cell(obj.TrainOptions)]',1,[]);
                
                %Train Model
                obj.LRmodel= fitlm(tbl,formula,nvPairs{:});
                
            end
            obj.UserData=obj.LRmodel;
        end
       
    end
end

