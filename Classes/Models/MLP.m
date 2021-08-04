classdef MLP < AutoRegressionModel
    %GP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        net
        TrainOptions struct
        layerSize
        parameters
    end
    
    methods
        function obj = MLP(Features,Targets,layerSize,varargin)
            %MLP Construct an instance of this class
            
            obj.Features=Features;
            obj.Targets=Targets;
            obj.layerSize=layerSize;
            
            p = inputParser();
            addParameter(p,'TrainFcn','trainlm',@ischar);
            addParameter(p,'DataSelection',@(tbl)tbl,@(x)isa(x,'function_handle'));
            parse(p,varargin{:})
            obj.parameters=p.Results;
        end
        
        
        function tbl=predict(obj,tbl)
            
            if isempty(obj.net)
                error("Model has not been trained before prediction")
            else
                x=tbl(:,obj.Features).Variables';
                pred=obj.net(x)';
                tbl(:,obj.Targets).Variables=pred;
                
            end
            
        end
        
        function obj = train(obj,tbl)
            %Select Trainingdata
            tbl=obj.parameters.DataSelection(tbl);
            
            %Specifiy Target and Features
            formula=obj.Targets+"~";
            for i=1:length(obj.Features)
                formula=formula+obj.Features(i);
                if i < length(obj.Features)
                    formula=formula+"+";
                end
            end
            
            %Train Model
            x=tbl(:,obj.Features).Variables';
            t=tbl(:,obj.Targets).Variables';
            
            if isempty(obj.net)
                obj.net= feedforwardnet(obj.layerSize,obj.parameters.TrainFcn);
            end
            
            obj.net=train(obj.net,x,t);
            
        end

    end
end


