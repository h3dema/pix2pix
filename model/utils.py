import os
import json
from datetime import datetime
import torch
from torch import nn


def initialize_weights(layer):
    """
    Initialize the weights of a given layer.

    Parameters:
    layer (torch.nn.Module): The layer whose weights are to be initialized. 
                             This can be an instance of nn.Conv2d, nn.ConvTranspose2d, 
                             nn.BatchNorm2d, or nn.InstanceNorm2d.

    Returns:
    None
    """
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif isinstance(layer, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0.0)
    return None


class Logger():
    """logger class for saving training logs"""

    def __init__(self,
                 exp_name: str='./runs',
                 filename: str=None):
        """
        Initializes the class with the given experiment name and filename.

        Args:
            exp_name (str): The name of the experiment directory. Defaults to './runs'.
            filename (str, optional): The name of the file. If None, the current date and time will be used. Defaults to None.

        Attributes:
            exp_name (str): The name of the experiment directory.
            cache (dict): A dictionary to store cached data.
            date (str): The current date and time formatted as "Month_Day_Year_Hour_MinuteAM/PM".
            filename (str): The name of the file, which is either the provided filename or the current date and time.
        """
        self.exp_name=exp_name
        self.cache={}
        if not os.path.exists(exp_name):
            os.makedirs(exp_name, exist_ok=True)
        self.date=datetime.today().strftime("%B_%d_%Y_%I_%M%p")
        if filename is None:
            self.filename=self.date
        else:
            self.filename="_".join([self.date, filename])
        fpath = f"{self.exp_name}/{self.filename}.json"
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        
    def add_scalar(self, key: str, value: float, t: int):
        """
        Adds a scalar value to the cache with a specified key and timestamp.

        Args:
            key (str): The key under which the scalar value is stored.
            value (float): The scalar value to be added.
            t (int): The timestamp associated with the scalar value.

        Returns:
            None
        """
        if key in self.cache:
            self.cache[key][t] = value
        else:
            self.cache[key] = {t:value}
        self.update()
        return None
    
    def save_weights(self, state_dict, model_name: str='model'):
        """
        Save the model weights to a file.

        Args:
            state_dict (dict): A dictionary containing the model's state.
            model_name (str, optional): The name to use for the saved model file. Defaults to 'model'.

        Returns:
            None
        """
        fpath = f"{self.exp_name}/{model_name}.pt"
        torch.save(state_dict, fpath)
        return None
    
    def update(self,):
        """
        Updates the JSON file with the current cache data.

        This method serializes the cache data to a JSON formatted string and writes it to a file 
        specified by the combination of `exp_name` and `filename` attributes.

        Returns:
            None
        """
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        return None
    
    def close(self,):
        """
        Closes the current session by writing the cached data to a JSON file and clearing the cache.

        The method constructs the file path using the experiment name and filename attributes,
        writes the cached data to the file in JSON format, and then clears the cache.

        Returns:
            None
        """
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        self.cache={}
        return None
    