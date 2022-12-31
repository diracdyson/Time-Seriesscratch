import numpy as np
import pmdarima as pm
import pandas as pd
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape
import numpy as np
import pmdarima as pm
import pandas as pd
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape
import matploblib.pyplot as plt
def forecast_one_step(model):
    fc, conf_int = model.predict(n_periods= ,return_conf_int=True)
    return(fc.tolist()[0],
          np.array(conf_int).tolist()[0])

# ensure y_train, y_test is still in dataframe object not nparray
def plot_results(y_train,y_test,forecastres,confidence_inter):
    fig, axes= plt.subplots(2, 1, figsize=(12,12))
    
    axes[0].plot(y_train, color='blue',label='Training Data')
    axes[0].plot(y_test.index, forecastres, color='green',marker='o',label='predicted price')
    axes[0].plot(y_test.index, y_test,color='red',label='Actual Price')
    
    axes[0].set_title('Hedge Fund Returns Prediction')
    axes[0].set_xlabel('Dates')
    axes[0].set_ylabel('Prices')
    
    axes[0].legend()
    # Conf int plot
    
    axes[1].plot(y_train,color='blue',label='Training Data')
    axes[1].plot(y_test.index,forecastres,color='green',label='Predicted Returns')
    
    axes[1].set_title('Returns Predictions & Confidence Intervals')
    axes[1].set_xlabel('Dates')
    axes[1].set_ylabel('Prices')
    
    axes[1].fill_between(y_test.index,confidence_inter[:,0],confidence_inter[:,1],
                        alpha=0.9, color='orange', label='confidence intervals')
    axes[1].legend()
    
    plt.show()
                
                 
def test_steps_forecast(y_train,y_test,model):
    forecastres=[]
    confidence_inter=[]
    for new_ob in y_test:
        
        fc, conf= forecast_one_step(model)
        forecastres.append(fc)
        confidence_inter.append(conf)
        
        model.update(new_ob)
        
    forecastres=np.array(forecastres)
    confidence_inter=np.array(confidence_inter)
    plot_results(y_train,y_test,forecastres,confidence_inter)
    print(f"Mean squared error: {mean_squared_error(y_test, forecastres)}")
    print(f"SMAPE: {smape(y_test, forecastres)}")
    # model summary will not display with pyplot above
    #
    return forecastres,confidence_inter

