classdef GP < AutoRegressionModel
    %GP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        GPRmodels
        TrainOptions struct
        WilkinsonNotations string
    end
    
    methods
        function obj = GP(Features,Targets,TrainOptions,DataSelection)
            %GP Construct an instance of this class
            %   Detailed explanation goes here
            
            obj.Features = Features;
            
            obj.Targets = Targets;
            
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
            
            %Specifiy Target and Features
            for j=1:length(obj.Targets)
                formula=obj.Targets(j)+"~";
                for i=1:length(obj.Features)
                    formula=formula+obj.Features(i);
                    if i < length(obj.Features)
                        formula=formula+"+";
                    end
                end
                obj.WilkinsonNotations(j)=formula;
            end
            
        end
        
        function tbl=predict(obj,tbl)
            
            if isempty(obj.GPRmodels)
                error("Model has not been trained before prediction")
            else
                for i=1:length(obj.GPRmodels)
                    tbl.(obj.Targets(i))=predict(obj.GPRmodels{i},tbl);       
                end
            end
            
        end
        
        function obj = train(obj,tbl)
            %Select Trainingdata
            tbl=obj.DataSelection(tbl);
            
            
            for i=1:length(obj.Targets)
                %Prepare Name Value Pairs
                if isempty(obj.TrainOptions)
                    obj.GPRmodels{i}= fitrgp(tbl,obj.WilkinsonNotations(i));
                else
                    nvPairs = reshape([fieldnames(obj.TrainOptions),...
                        struct2cell(obj.TrainOptions)]',1,[]);
                    
                    %Train Model
                    obj.GPRmodels{i}= fitrgp(tbl,obj.WilkinsonNotations(i),nvPairs{:});
                end
            end
            %obj.UserData=obj.GPRmodels;
        end
    end
end

