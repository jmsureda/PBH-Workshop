
parameters = {"Ms":1e-1,"nb":2.,"k_piv":10.,"fm":1.,"An":None,"a_FCT":None} # Default parameters

def check_parameters(**kwargs):
    
    '''
    Checks if the given parameters exists and fills the dictionary with 
    the default values for the parameters that were not given.
    '''
    
    #Checks if the given parameters exist.
    for i in kwargs.keys():
        
        if i not in parameters.keys():
            
            raise ValueError("%s is not a parameter."%i)
        
    # If some parameter is not given, it uses the default value.
    
    for i in parameters.keys():
        
        if i not in kwargs.keys():
            
            kwargs[i] = parameters[i]
            
    return kwargs

### Move this to each mass function. Then the massfunction class does not do the check directly. 

## De esta manera, cada tipo de función de masa dependerá de logM + **kwargs y queda super standard :)