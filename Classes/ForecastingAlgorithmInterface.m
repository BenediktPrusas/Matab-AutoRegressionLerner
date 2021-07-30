classdef ForecastingAlgorithmInterface < handle
    
    %ForecastingAlgorithmInterface Subclasses can implement Algorithms, which can be used with ClosedLoop Bechmark 
    
    properties
        Features string %String Array of the Variables names that will be used in Tables
        Targets string% String Array of the Variables names that will be used in Tables
        UserData cell %Specific Information which should be tracked during testig e.g. HyperparameterOptimatsation Results
       
    end
    
    methods (Abstract)
      predict(obj,ttbl)
      % Predicts Targets based on Features in the timetable
      % Targets and Features are specified in the obj Properties
      predictCL(obj,ttbl)
      % Close Loop predicts all Targets, which value is NaN based on Features in the timetable
      % Targets and Features are specified in the obj Properties
      train(obj,ttbl)
      % Trains models on the given timetable Data
   end
    
end

