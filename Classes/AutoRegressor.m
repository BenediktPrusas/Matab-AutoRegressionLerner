classdef AutoRegressor
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess='private')
        Models
        TrainingOptions
        ExogenousVariables string
        All_targets string
        PredictionOrder
    end
    
    
    methods
        function obj = AutoRegressor(varargin)
            %AutoRegressor Construct an instance of this class
            %   Detailed explanation goes here
            if nargin>0
                obj.addModel(varargin{1});
            end
        end
        
        function tbl = predict(obj,tbl)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            
            tbl=obj.feature2target(tbl); 
            %iterate over rows with missing target values
            [rows,~] = find(ismissing(tbl(:,obj.All_targets)));
            rows=unique(rows)';
            for row=rows
                %Let each Model Predict
                for a=obj.PredictionOrder
                    tbl(row,:)=obj.Models{a}.predict(tbl(row,:));
                end
                
                % fill in for the next timestep
                if row < height(tbl)
                charArr=char(obj.All_targets);
                for j=1:size(charArr,3)
                    if (isequal(charArr(1,1:5,j),'Next_'))
                        charArr(1,1:5,j)=' ';
                        feature_name=strtrim(string(charArr(1,:,j)));
                        tbl.(feature_name)(row+1)=tbl.(obj.All_targets(j))(row);
                    end
                end 
                end
            end
            % Remove 'Next_' Columts
            tbl(:,obj.All_targets)=[];
            
        end
        
        function obj = addModel(obj,UserIn)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            
            %check for cell array
            %check superclass
            %Add Model to List
            %Update Features and Targets
            %update prediction order
            
            %addAlgorithm Adds an Algoritm or cell Array of Algorithms to
            %the benchmark object
            s=superclasses(UserIn);
            if ~isempty(s) && s{1}=="AutoRegressionModel"
                obj.Models{end+1}=UserIn;
                obj.All_targets=[obj.All_targets,UserIn.Targets];
                %obj.All_features=union(obj.All_features,UserIn.Features);
                obj.PredictionOrder=obj.determinePredictionOrder;
            else
                %case when a cell array of algorithms is delivered
                if class(UserIn)=='cell'
                    if length(size(UserIn))==2 && (sum(size(UserIn)==1)>0)
                        for i=1:length(UserIn)
                            s=superclasses(UserIn{i});
                            if ~isempty(s) &&s{1}=="AutoRegressionModel"
                                obj.Models{end+1}=UserIn{i};
                                obj.All_targets(end+1)=UserIn{i}.Targets;
                                obj.All_features=union(obj.All_features,UserIn{i}.Features);
                            else
                                error("Algorithm object must have AutoRegressionModel as superclass")
                            end
                        end
                        obj.PredictionOrder=obj.determinePredictionOrder;
                    end
                end
            end
        end
        
        function obj = train(obj,tbl)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            tbl=obj.feature2target(tbl);
            for i=1:numel(obj.Models)
                obj.Models{i}=obj.Models{i}.train(tbl);
                
            end
        end
        
        function info(obj)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            
            % print variables
            % types and numbers of models
            % dependencys
            
        end
        
    end
    methods (Hidden=true, Access = 'private')
        
        function PredictionOrder = determinePredictionOrder(obj)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            PredictionOrder=[];
            
            %contruct graph
            
            if numel(obj.All_targets)<numel(obj.Models)
                error("Some Target gets predicted by more then one Algorithm")
            end
            
            graph=zeros(length(obj.Models));
            
            for i=1:length(obj.Models)
               for j=1:length(obj.Models) 
                dependent=intersect(obj.Models{i}.Features,obj.Models{j}.Targets);

                
                if isempty(dependent)
                    graph(i,j)=0;
                else
                    graph(i,j)=1;
                end
               end
            end
            %iterate over graph
            if sum(graph,'all') >0
                if sum(diag(graph))>0
                    error("A Model prediction depends on it self")
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
                            PredictionOrder=[PredictionOrder,setdiff(1:length(obj.Models),PredictionOrder)];
                            break;
                        else
                            error("Dependencies cannot be resolved")
                        end
                    end
                end
            end
        end
        
        function tbl = feature2target(obj,tbl)
            charArr=char(obj.All_targets);
            for j=1:size(charArr,3)
                if (isequal(charArr(1,1:5,j),'Next_'))
                    charArr(1,1:5,j)=' ';
                    feature_name(j)=strtrim(string(charArr(1,:,j)));
                end
            end
            for i=1:length(feature_name)
                temp=tbl.(feature_name(i));
                tbl.("Next_"+feature_name(i))=[temp(2:end); NaN];
            end
            tbl(end,:)=[];
            
        end
        
        function names=trimNext_(Array)
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

