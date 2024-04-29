import numpy as np

'''
The univariate portfolio analysis procedure has four steps.
 - The first step is to calculate the breakpoints that will be used to divide the sample into portfolios. 
 - The second step is to use these breakpoints to form the portfolios. 
 - The third step is to calculate the average value of the outcome variable Y within each portfolio for each period t.
 - The fourth step is to examine variation in these average values of Y across the different portfolios.

'''

class portfolio_analysis():
    def __init__(self) -> None:
         pass
    
    def breakpoint(self, character: np.ndarray, number: int=None, percentitles: list[int]=None) -> np.ndarray:
        '''
        Calculate percentiles as breakpoints based on the given feature array and list of percentiles.

        Args:
        character (np.ndarray): Feature array used for calculating percentiles.
        number (int, optional): Number of percentiles to generate. Default is None, in which case the default 100 percentiles will be used.e.g.[0, 20, 40, 60, 80, 100]
        percentitles (list[int], optional): Custom list of percentiles. Default is None, in which case an evenly spaced list of percentiles will be used.e.g.[0, 20, 40, 60, 80, 100]
    
    Returns:
        np.ndarray: Array of percentiles as breakpoints.

    Note:
        If a custom list of percentiles is not provided, an evenly spaced list of percentiles will be used by default, including 0 and 100.e.g.[0, 20, 40, 60, 80, 100]
        '''
        if percentitles is None:
            percentitles = np.linspace(0, 100, number+2, dtype=int)
            if number is None:
                percentitles = np.linspace(0, 100, 6, dtype=int)
                # [0, 20, 40, 60, 80, 100]
        else:
            if percentitles[0] != 0:
                percentitles.insert(0, 0)
            if percentitles[-1] != 100:
                percentitles.append(100)
            percentitles = percentitles
        
        breakpoint = np.percentile(character, percentitles)

        return breakpoint
    

    def portfolio_formation(self, character, breakpoint)->np.ndarray:
        '''
        Form portfolios based on given characteristics and breakpoints.

        Args:
            character (np.ndarray): Array of characteristics for each sample.
            breakpoint (list[float]): List of breakpoints for forming portfolios.

        Returns:
            np.ndarray: Array representing the membership of each sample in different portfolios.

        Note:
            The function assigns samples to portfolios based on the given breakpoints. Each sample's characteristic is compared with the breakpoints to determine its membership in the portfolios. The function returns an array where each row represents a sample and each column represents a portfolio. The value at each position indicates the membership of the sample in the corresponding portfolio.
        '''
        n = len(character)     # Number of samples
        p = len(breakpoint) - 1 # Number of portfolios

        label = np.zeros((n, p))

        for i in range(p):
            label[:, i][np.where((character >= breakpoint[i]) & (character <= breakpoint[i+1]))] = i + 1

        return label

    



         
         