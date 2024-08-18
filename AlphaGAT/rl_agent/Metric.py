
import numpy as np

##metric to evaluate methods
class Metric:
    def __init__(self, return_array) -> None:
        ###return array shape[day_length]  [1.1, 0,9, 1.025, .....]
        self.return_array = np.array(return_array)
        self.days_in_year = 252
        self.risk_free_rate = 0

    def cal_cumulative_returns(self):
        self.cum_array = np.cumprod(self.return_array)
        self.CW = self.cum_array[-1]

    def cal_apy(self):
        self.APY = (self.CW) ** (self.days_in_year / len(self.return_array)) - 1


    def cal_asr(self):
        self.ASR = (self.APY - self.risk_free_rate) / np.std(self.return_array - 1) * np.sqrt(self.days_in_year)


    def cal_cr(self):

        drawdowns = []
        for i in range(len(self.cum_array)):
            max_array = max(self.cum_array[:i + 1])
            drawdown = max_array - self.cum_array[i]
            drawdowns.append(drawdown)
        MD = max(drawdowns)
        self.CR = self.APY / MD

    def calculate_print(self):
        self.cal_cumulative_returns()
        self.cal_apy()
        self.cal_asr()
        self.cal_cr()
        # 打印结果
        print("Cumulative wealth:{:.4}".format(self.CW))
        print("Annualized Percentage Yield (APY): {:.4}".format(self.APY))
        print("Annualized Sharpe Ratio (ASR): {:.4f}".format(self.ASR))
        print("Calmar Ratio: {:.4f}".format(self.CR))