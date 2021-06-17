from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(context='notebook', palette='Reds_r')

import numpy as np
import pandas as pd
from easydict import EasyDict as edict


from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, confusion_matrix, classification_report


@dataclass
class MultilabelTrainer:
    
    cfg: edict
    model_class: Optional[class]
    X_train: np.ndarray
    y_train: np.ndarray
    idx2column: None
    verbose: bool = False
    kwargs = dict()
    
        
    def __post_init__(self):
        
        self.num_class = self.cfg.num_class
        self.models = dict()
       

    def run_all(self, X_train=None, y_train=None, X_test=None, y_test=None):
        
        if X_train is None:
            X_train = self.X_train
            
        if y_train is None:
            y_train = self.y_train
        
        if X_test is None or y_test is None:
            print('Validation set should be given')
            return
            
        self.fit()
        self.predict(X_test=X_test)
        self.get_auc(target=y_test)
        self.get_acc(target=y_test)
        self.chance_level()
        self.export_df()        

        
    def fit(self):
        
        for c in range(self.num_class):
            
            if self.idx2column is not None and self.verbose:
                print(f'Fit {self.idx2column[c].upper()}.')
            _model = self.model_class(**self.kwargs)
            _model.fit(self.X_train, self.y_train[:, c])
            self.models[c] = _model
            
            
    def predict(self, X_test):
        
        self.result = dict()
        for c in range(self.num_class):
            
            if self.idx2column is not None and self.verbose:
                print(f'Predict {self.idx2column[c].upper()}.')
            _y = self.models[c].predict(X_test)
            self.result[c] = _y
            
        return self.result
    
    
    def get_auc(self, target, pred=None):
        
        '''
        target: y_test (num_valid, num_class)
        pred  : in the form of dict with class: pred (num_valid,)
            - will be parsed into (num_valid, num_class) of ndarray
            - if not given, it will use the previous result saved in 
        
        Calculates both
        - overall AUC -> self.overall_auc
        - each class AUC -> self.auc
        '''
        
        self.auc = dict()
        if pred is None or isinstance(pred, dict):
            if hasattr(self, 'result'):
                pred = np.column_stack([*self.result.values()])            
            
            else:
                print('Please infer with valid/test data first.')
                
            
        for c in range(self.num_class):            
            self.auc[self.idx2column[c]] = roc_auc_score(target[:, c], pred[:, c])

        self.overall_auc = roc_auc_score(target, pred, multi_class='ovr')
        return self.overall_auc
    
    
    def get_acc(self, target, pred=None):
        
        '''
        Same with get_auc except that it does NOT return AUC
        '''
        
        self.acc = dict()
        if pred is None or isinstance(pred, dict):
            pred = np.column_stack([*self.result.values()])
            
        for c in range(self.num_class):
            self.acc[self.idx2column[c]] = accuracy_score(target[:, c], pred[:, c])
            
        return self.acc
    

    def chance_level(self, target=None):
        
        '''
        Calculate how many instances were in each class
        If target is not given, it will automatically use y_train.
        '''
        
        if target is None:
            target = self.y_train
            
        self.chance_level = dict()
        for c in range(self.num_class):
            self.chance_level[self.idx2column[c]] = target[:, c].sum() / len(target)
        
        return self.chance_level
    
    
    def export_df(self):
        
        '''
        possible feats
        'idx2column', 'result', 'auc', 'acc', 'chance_level'
        '''
        
        df = dict()
        self.feats = ['auc', 'acc', 'chance_level']
        for feat in self.feats:
            
            if hasattr(self, feat):
                df[feat] = getattr(self, feat)
        
        self.df = pd.DataFrame(df)
        return self.df
    
    
    def plot_acc_proportion(self, feats: list =None):
        
        if feats is None:
            feats = ['acc', 'chance_level']
            
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f'{self.model_class.__name__} Chance Level vs. Accuracy', size='large')
        ax.set_xlabel('Label Class', size='large')
        ax.set_ylabel('Proportion (%)', size='large')
        if not hasattr(self, 'df'):
            self.export_df()
        self.df[feats].plot(kind='bar', ax=ax)
        
        return ax