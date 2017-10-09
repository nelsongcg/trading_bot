#The present code is the first step in creating a trading bot, at the present moment it searches for the best buy and
#sell oppotunities given the Bollinger Bands strategy

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D


class predictor:
    def __init__(self,conn,period,pairs,startTime,endTime):
        self.populationSize = 100
        self.conn = conn
        self.period = period
        self.pairs = pairs
        self.startTime = startTime
        self.endTime = endTime
        self.population = pd.DataFrame()
        self.fitnessGraph = []
        self.maxFitnessGraph = []

    #Funcion that calculates bollinger bands
    def getBollingerBands(self,data,window,stdRange):

        rollingMean = data.rolling(window=window, center=False).mean()
        rollingStd = data.rolling(window=window, center=False).std()
        upperBand = rollingMean + ( rollingStd* stdRange)
        lowerBand = rollingMean - ( rollingStd* stdRange)

        rollingMean.columns = ["RollingMean"]
        upperBand.columns = ["UpperBand"]
        lowerBand.columns = ["LowerBand"]

        return [rollingMean,upperBand,lowerBand,rollingStd]

    #Function that returns the fitness value given BB parameters
    def getFitness(self,data,window,stdRange):

        rollingMean, upperBand, lowerBand, rollingStd = self.getBollingerBands(data,window,stdRange)


        lowerBandArray = np.array(lowerBand[["LowerBand"]])[window - 1:]
        upperBandArray = np.array(upperBand[["UpperBand"]])[window - 1:]
        price = np.array(data[['weightedAverage']])[window - 1:]
        overUpperBand = price >= upperBandArray
        underLowerBand = price <= lowerBandArray

        buypositions = underLowerBand[1:] < underLowerBand[:-1]
        sellpositions = overUpperBand[1:] < overUpperBand[:-1]

        buy = []
        sell = []
        holdingCoin = False
        price = price[1:]
        for i in range(len(price)):
            if holdingCoin:
                if sellpositions[i][0] == True:
                    sell.append(price[i][0])
                    holdingCoin = False
            else:
                if buypositions[i][0] == True:
                    buy.append(price[i][0])
                    holdingCoin = True


        if (len(buy) > len(sell)):
            buy = buy[:-1]


        buy = np.array(buy)
        sell = np.array(sell)
        holdingCoin = True
        bitcoin = 1.0
        money = 0
        for c in range(len(sell)):
            if holdingCoin:
                money = sell[c]*bitcoin
                holdingCoin = False
            else:
                bitcoin = money/buy[c]
                holdingCoin = True

        profitLoss = sell - buy

        percentProfitLoss = (money - buy[0])/buy[0]
        stdProfitLoss = np.std(profitLoss)
        if len(sell) == 0:
            percentWinners = 0
        else:
            numTrades = len(sell)
            percentWinners = float(sum(profitLoss > 0)) / numTrades

        percentWinners = float(sum(profitLoss > 0))/numTrades
        fitness = percentProfitLoss*percentWinners
        buyNHold = (price[-1] - price[0]) / price[0]
        #print "Fitness: ", fitness, "#Trades: ", len(sell), "Window: ", window, "STD: ", stdRange, "Profit: ",percentProfitLoss, "#Winners: ", percentWinners,"benchmark: ", buyNHold
        return fitness

    #Functions that randomly runs inside crossover in order to produce mutant individuals
    def mutation(self,window1,std1,window2,std2):

        window1 = int(random.uniform(0.5,1.5)*window1)
        window2 = int(random.uniform(0.8,1.2)*window2)
        std1 = random.uniform(0.8, 1.2) * std1
        std2 = random.uniform(0.8, 1.2) * std2

        return [window1,std1,window2,std2]

    #Combine two individuals in order to produce two new offspring
    def crossover(self,data,population):

        ind1Index = np.random.randint(0,self.populationSize)
        ind2Index = np.random.randint(0,self.populationSize)

        ind1 = population.ix[ind1Index]
        ind2 = population.ix[ind2Index]
        window1 = int(ind1.values[0])
        window2 = int(ind2.values[0])
        std1 = ind2.values[1]
        std2 = ind1.values[1]

        if random.randint(0,9) <= 1:
            window1,std1,window2,std2 = self.mutation(window1,std1,window2,std2)

        fitnessChild1 = self.getFitness(data, window1, std2)
        fitnessChild2 = self.getFitness(data, window2, std1)

        worstFitness = population.ix[self.populationSize-1,["fitness"]].values[0]

        if worstFitness < fitnessChild1:
            population = population.drop(self.populationSize-1)
            a = pd.DataFrame([[ind1.values[0], ind2.values[1], fitnessChild1]], columns=["Window", "STD", "fitness"])
            population = population.append(a)
            population = population.sort(["fitness"], ascending=False)
            population = population.reset_index(drop=True)
            worstFitness = population.ix[self.populationSize - 1, ["fitness"]].values[0]

        if worstFitness < fitnessChild2:
            population = population.drop(self.populationSize - 1)
            a = pd.DataFrame([[ind2.values[0], ind1.values[1], fitnessChild2]], columns=["Window", "STD", "fitness"])
            population = population.append(a)
            population = population.sort(["fitness"], ascending=False)
            population = population.reset_index(drop=True)

        return population

    #Functions that gathers all the information and plots the solution
    def getOrderTrigger(self):

        historicalData = self.conn.api_query("returnChartData",
                                                  {"currencyPair": self.pairs[0], "start": self.startTime, "end": self.endTime,
                                                   "period": self.period})
        historicalData = pd.DataFrame(historicalData)[['weightedAverage']]

        # Generate initial population
        window = np.random.randint(1,2000,size=self.populationSize)
        STD = np.random.uniform(1,3,size=self.populationSize)

        fitness = []
        for i in range(self.populationSize):
            fitness.append(self.getFitness(historicalData,window[i],STD[i]))


        self.population["Window"] = window
        self.population["STD"] = STD
        self.population["fitness"] = fitness

        self.population = self.population.sort(["fitness"],ascending=False)
        self.population = self.population.reset_index(drop=True)

        fig = plt.figure()
        ax = fig.add_subplot(122, projection='3d')


        ax2 = fig.add_subplot(1, 2, 1)

        #Functions used to produce a live graph
        def update(i,historicalData,population):

            self.population = self.crossover(historicalData,self.population)
            self.fitnessGraph.append(np.mean(self.population.ix[:,["fitness"]]).values[0])
            self.maxFitnessGraph.append(np.max(self.population.ix[:,["fitness"]]).values[0])
            axes = plt.gca()
            axes.set_xlim([0, 2000])
            axes.set_ylim([0, 3])

            ax2.clear()
            ax.clear()

            ax.set_title('Population')
            ax.set_xlabel("Window")
            ax.set_ylabel("Standard deviation")
            ax.set_zlabel("Fitness")

            ax2.set_title('Fitness evolution')
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Average Fitness")

            ax.scatter(self.population.ix[:,["Window"]],self.population.ix[:,["STD"]],self.population.ix[:,["fitness"]])
            ax2.plot(self.fitnessGraph)

        a = anim.FuncAnimation(fig,update,fargs=(historicalData,self.population),frames=1000,repeat=False)
        plt.show()

