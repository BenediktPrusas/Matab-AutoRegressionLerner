classdef GP < AutoRegressionModel
    %GP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        GPRmodel
        TrainOptions struct
    end
    
    methods
        function obj = GP(Features,Targets,TrainOptions,DataSelection)
            %GP Construct an instance of this class
            %   Detailed explanation goes here
            
            obj.Features = Features;
            
            if length(Targets)==1
            obj.Targets = Targets;
            else
                error("This implemtation of Gaussian Processes can only have one target")
            end
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

            if isempty(obj.GPRmodel)
                error("Model has not been trained before prediction")
            else
                pred=obj.GPRmodel.predict(tbl);
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
                obj.GPRmodel= fitrgp(tbl,formula);
            else
                nvPairs = reshape([fieldnames(obj.TrainOptions),...
                    struct2cell(obj.TrainOptions)]',1,[]);
                
                %Train Model
                obj.GPRmodel= fitrgp(tbl,formula,nvPairs{:});
            end
            obj.UserData=obj.GPRmodel;
        end
    end
end

