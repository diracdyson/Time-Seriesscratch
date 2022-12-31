import numpy as np
import pmdarima as pm
import pandas as pd

def forecast_one_step(model):
    fc, conf_int = model.predict(n_periods= ,return_conf_int=True)
    return(fc.tolist()[0],
          np.array(conf_int).tolist()[0])


def test_steps_forecast(y_test,model):
    forecastres=[]
    confidence_inter=[]
    for new_ob in y_test:
        
        fc, conf= forecast_one_step(model)
        forecastres.append(fc)
        confidence_inter.append(conf)
        
        model.update(new_ob)
        
    forecastres=np.array(forecastres)
    confidence_inter=np.array(confidence_inter)
    print(f"Mean squared error: {mean_squared_error(y_test, forecastres)}")
    print(f"SMAPE: {smape(y_test, forecastres)}")
    return forecastres,confidence_inter

