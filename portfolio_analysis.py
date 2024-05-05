import numpy as np
import pandas as pd
'''
The univariate portfolio analysis procedure has four steps.
 - The first step is to calculate the breakpoints that will be used to divide the sample into portfolios. 
 - The second step is to use these breakpoints to form the portfolios. 
 - The third step is to calculate the average value of the outcome variable Y within each portfolio for each period t.
 - The fourth step is to examine variation in these average values of Y across the different portfolios.

'''

class PortfolioAnalysis():
    def __init__(self, dataframe: pd.DataFrame):
        # 判断传入的数据是否为 DataFrame
        if type(dataframe) is not pd.DataFrame:
            raise ValueError("Not DataFrame format")
        
        self.df = dataframe  # initial dataframe
    
    def breakpoint(self, feature_percentiles: dict[str, list[int]]) -> dict[str, np.ndarray]:
        '''
        The first step is to calculate the breakpoints that will be used to divide the sample into portfolios.
        Calculate breakpoints for each feature based on the given percentiles.

        Args:
            feature_percentiles (dict): A dictionary containing column names as keys and corresponding percentile lists as values.

            Example:
            feature_percentiles = {
                'feature1': [10, 30, 50, 70, 90],
                'feature2': [20, 40, 60, 80]
            }

        Returns:
            dict: A dictionary containing column names as keys and arrays of percentiles as breakpoints as values.

        Raises:
            ValueError: If the provided percentiles are not in list format.
        '''

        breakpoints_dict = {}

        for feature, percentiles_list in feature_percentiles.items():
            if not isinstance(percentiles_list, list):
                raise ValueError("The provided percentiles are not in list format.")
            
            percentiles_list = sorted(percentiles_list)
            if percentiles_list[0] == 0:
                percentiles_list.pop(0)
            if percentiles_list[-1] == 100:
                percentiles_list.pop(-1)

            breakpoints_dict[feature] = np.percentile(self.dataframe[feature].values, percentiles_list)
            # Not including zero and 100%

        return breakpoints_dict




    def portfolio_formation(self, breakpoints_dict: dict[str, np.ndarray], const: str='bi') -> np.ndarray:
        '''
            The second step is to use these breakpoints to form the portfolios.
            Generate multivariate portfolios based on breakpoints for each feature.
        Args:
            breakpoints_dict (dict): A dictionary where keys represent feature names and values represent lists of breakpoints for each feature.
            const (str, optional): Type of portfolios to generate. Defaults to 'univariate'. Another is bivariate
        
        Returns:
            np.ndarray: An array where each row represents a sample and each column represents a portfolio label.

        Note:
            This function generates portfolios based on breakpoints provided for each feature. It computes labels for each sample based on these breakpoints and returns an array of portfolio labels.
        '''

        for feature, breakpoint in breakpoints_dict.items():
            bins = sorted(breakpoint + [-np.inf, np.inf])
            self.df[feature+'_group'] = pd.cut(self.df[feature], bins=bins, labels=False, right=True)

        if const == 'uni':
            self.df['portfolio'] = self.df[list(breakpoints_dict.keys())[0]+'_group']  
        if const == 'bi':
            self.df['portfolio'] = list(zip(self.df[list(breakpoints_dict.keys())[0]+'_group'], self.df[list(breakpoints_dict.keys())[1]+'_group']))

        return self.df['portfolio'].values

    def average_portfolio_values(self, portfolio_label: np.ndarray, outcome: np.ndarray, weight: np.ndarray=None):
        """
        Calculate the average values of portfolios.

        Args:
            portfolio_label (np.ndarray): Array of portfolio labels, where each column represents labels for a portfolio.
            outcome (np.ndarray): Array of outcome values for each portfolio corresponding to portfolio_label.
            weight (np.ndarray, optional): Array of weights used for computing weighted averages. Defaults to None, representing equal weights.

        Returns: (turple): 
            average_outcome (np.ndarray): Array containing the average values of each portfolio, with each row representing the average value of a portfolio.
            HML_average_outcome (np.ndarry): Difference in the average outcome variable between the highest and lowest portfolios.

        Note:
            If no weights are provided (weight is None), equal-weighted portfolios are computed.
        """
        p = portfolio_label.shape[1]
        average_outcome = np.zeros((p, 1))

        for i in range(p):
            if weight is None: # equal-weighted portfolios.
                average_outcome[i, 0] = np.mean(outcome[np.where(portfolio_label[:, i] == i+1)])
            else:              # weighted portfolios
                average_outcome[i, 0] = np.average(outcome[np.where(portfolio_label[:, i] == i+1)], weights=weight)

        HML_average_outcome = average_outcome[-1] - average_outcome[0]

        return (average_outcome, HML_average_outcome)



         
         