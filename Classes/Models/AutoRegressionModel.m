classdef AutoRegressionModel
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Features string %String Array of the Variables names that will be used in Tables
        Targets string% String Array of the Variables names that will be used in Tables
        DataSelection
        UserData
    end
    
    methods (Abstract)
      predict(obj,tbl)
      % Predicts Targets based on Features in the timetable
      % Targets and Features are specified in the obj Properties
      train(obj,tbl)
      % Trains models on the given timetable Data
   end
end

