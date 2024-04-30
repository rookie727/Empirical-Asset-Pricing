import numpy as np
import pandas as pd
import itertools
'''
The univariate portfolio analysis procedure has four steps.
 - The first step is to calculate the breakpoints that will be used to divide the sample into portfolios. 
 - The second step is to use these breakpoints to form the portfolios. 
 - The third step is to calculate the average value of the outcome variable Y within each portfolio for each period t.
 - The fourth step is to examine variation in these average values of Y across the different portfolios.

'''

class PortfolioAnalysis():
    def __init__(self, dataframe: pd.DataFrame) -> None:
        # 判断传入的数据是否为 DataFrame
        if type(dataframe) is not pd.DataFrame:
            raise ValueError("传入的数据不是 DataFrame 格式")
        
        self.dataframe = dataframe  # 初始化数据
    
    def breakpoint(self, feature_percentiles: dict[str, list[int]]) -> dict[str, np.ndarray]:
        '''
        Calculate breakpoints for each feature based on the given percentiles.

        Args:
            percentiles (dict[str, list[int]]): A dictionary containing column names as keys and corresponding percentile lists as values.

        Returns:
            breakpoints (dict[str, np.ndarray]): A dictionary containing column names as keys and arrays of percentiles as breakpoints as values.

        Raises:
            ValueError: If the provided percentiles are not in list format.
        '''
        breakpoints_dict = {}

        for feature, percentiles_list in feature_percentiles.items():
            if not isinstance(percentiles_list, list):
                raise ValueError("百分位数必须是列表格式")
            
            percentiles_list = sorted(percentiles_list)
            if percentiles_list[0] != 0:
                percentiles_list.insert(0, 0)
            if percentiles_list[-1] != 100:
                percentiles_list.append(100)

            breakpoints_dict[feature] = np.percentile(self.dataframe[feature].values, percentiles_list)

        return breakpoints_dict


    def portfolio_formation(self, breakpoints_dict):
        '''
        Generate multivariate portfolios based on breakpoints for each feature.

        Args:
        breakpoints_dict (dict): A dictionary where keys represent feature names and values represent lists of breakpoints for each feature.
        
        Returns:
        np.ndarray: An array where each row represents a sample and each column represents a portfolio label.

        Note:
        This function generates portfolios based on breakpoints provided for each feature. It computes labels for each sample based on these breakpoints and returns an array of portfolio labels.
        '''
        n = self.dataframe.shape[0]  # Number of samples
        num_features = len(breakpoints_dict)  # Number of features
        num_breakpoints = [len(breakpoints) - 1 for breakpoints in breakpoints_dict.values()]  # Number of breakpoints for each feature; including 0% and 100%
        p = np.prod(num_breakpoints)  # Number of portfolios, e.g., [0, 30, 70, 100] -> 3 portfolios

        portfolio_labels = np.zeros((n, p))

        # Generate all possible combinations of breakpoints for all features
        breakpoints_combinations = np.array(list(itertools.product(*[range(i) for i in num_breakpoints])))

        # Convert breakpoints to numpy arrays for efficient broadcasting
        breakpoints_arrays = {feature: np.array(breakpoints) for feature, breakpoints in breakpoints_dict.items()}

        # Iterate through each combination of breakpoints
        for i, breakpoints_tuple in enumerate(breakpoints_combinations):
            # Get start and end breakpoints for each feature using broadcasting
            breakpoints_start = np.array([breakpoints_arrays[feature][breakpoint_index] for feature, breakpoint_index in zip(breakpoints_dict.keys(), breakpoints_tuple)])
            breakpoints_end = np.array([breakpoints_arrays[feature][breakpoint_index + 1] for feature, breakpoint_index in zip(breakpoints_dict.keys(), breakpoints_tuple)])

            # Compute labels for each sample based on breakpoints for all features
            temp_labels = np.sum((self.dataframe.values >= breakpoints_start) & (self.dataframe.values <= breakpoints_end), axis=1)

            # Assign the temporary labels to the corresponding column in the portfolio_labels array
            portfolio_labels[:, i] = temp_labels

        return portfolio_labels


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



         
         