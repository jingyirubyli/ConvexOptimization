import numpy as np
import copy
import pandas as pd
from gurobipy import *
import matplotlib.pyplot as plt
import time
import re

class Data:
    def __init__(self):
        self.nodeNum = 0
        self.N_lst = []

        self.corX = []
        self.corY = []
        self.costMatrix = None

        self.corX_Drone = []
        self.corY_Drone = []

        self.corX_Truck = []
        self.corY_Truck = []

        self.N_outNum = 0
        self.N_out_lst = []

        self.N_inNum = 0
        self.N_in_lst = []

        self.customerNum = 0 
        self.C_lst = []

        self.DroneCustomerNum = 0
        self.C_Drone_lst = []

        self.sortiesNum = 0
        self.P_lst = []

def readData(file_path, customerNum, Drone_endurance):
    data = Data()
    data.customerNum = customerNum
    data_df = pd.read_csv(file_path)
    for i in range(0, data.customerNum+1):
        data.N_lst.append(data_df.loc[i, 'CUST NO']-1)
        data.corX.append(data_df.loc[i, 'XCOORD'])
        data.corY.append(data_df.loc[i, 'YCOORD'])
        if data_df.loc[i, 'DRONE'] == 1:
            data.C_Drone_lst.append(data_df.loc[i, 'CUST NO']-1)
            data.corX_Drone.append(data_df.loc[i, 'XCOORD'])
            data.corY_Drone.append(data_df.loc[i, 'YCOORD'])
        else:
            data.corX_Truck.append(data_df.loc[i, 'XCOORD'])
            data.corY_Truck.append(data_df.loc[i, 'YCOORD'])

    data.N_lst.append(customerNum+1)
    data.C_lst = copy.deepcopy(data.N_lst[1:-1])
    data.corX.append(data_df.loc[0, 'XCOORD'])
    data.corY.append(data_df.loc[0, 'YCOORD'])

    data.nodeNum = len(data.N_lst)
    data.DroneCustomerNum = len(data.C_Drone_lst)
    data.N_out_lst = copy.deepcopy(data.N_lst[0:data.nodeNum-1])
    data.N_outNum = len(data.N_out_lst)
    data.N_in_lst = copy.deepcopy(data.N_lst[1:data.nodeNum])
    data.N_inNum = len(data.N_in_lst)

    data.costMatrix = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(0, data.nodeNum):
        for j in range(0, data.nodeNum):
            if i != j:
                data.costMatrix[i][j] = ((data.corX[i]-data.corX[j])**2+(data.corY[i]-data.corY[j])**2)**0.5
            else:
                pass

    for i in data.N_out_lst:
        for j in data.C_Drone_lst:
            if i != j:
                for k in data.N_in_lst:
                    if k != j and k != i and data.costMatrix[i][j] + data.costMatrix[j][k] < Drone_endurance:
                        data.P_lst.append((i, j, k))
    data.sortiesNum = len(data.P_lst)
    print("costMatrix:")
    print(data.costMatrix)
    # print("costMatrix[4][1]:")
    # print(data.costMatrix[4][1])
    return data

class Solution:
    ObjVal = 0 
    X = None
    X_routes = None
    Y = None
    Y_routes = None
    X_routesNum = 0
    Y_routesNum = 0

    def __init__(self, data, model): 
        self.ObjVal = model.ObjVal
        self.X = [[0 for j in range(0, data.nodeNum)] for i in range(0, data.nodeNum)] 
        self.X_routes = []
        self.Y = [[[0 for k in range(0, data.nodeNum)] for j in range(0, data.nodeNum)] for i in range(0, data.nodeNum)]
        self.Y_routes = []
        self.U = [0 for k in range(0, data.nodeNum)]
        self.T = [0 for k in range(0, data.nodeNum)]
        self.TDrone = [0 for k in range(0, data.nodeNum)]


def getSolution(data, model): 
    solution = Solution(data, model)
    var_lst = model.getVars()
    print('var_lst:')
    for i in var_lst:
        if i.x !=0:
            print(i)
    for v in model.getVars():
        split_arr = re.split(r"_", v.VarName)
        if split_arr[0] == 'X' and round(v.x) != 0:
            # print(v)
            solution.X[int(split_arr[1])][int(split_arr[2])] = v.x  # X_ij
        elif split_arr[0] == 'Y' and round(v.x) != 0:
            solution.Y[int(split_arr[1])][int(split_arr[2])][int(split_arr[3])] = v.x  # Y_ijk
        elif split_arr[0] == 'U' and v.x != 0:
            solution.U[int(split_arr[1])] = v.x  # U_i
        elif split_arr[0] == 'T' and v.x != 0:
            solution.T[int(split_arr[1])] = v.x  # T_i
        elif split_arr[0] == 'TDrone' and v.x != 0:
            solution.TDrone[int(split_arr[1])] = v.x  # TDrone_i
        else:
            pass
            # print(v)
    print("The truck routing solution.X:", solution.X)
    print("The drone routing solution.Y:", solution.Y)
    print("The visit sequence of the customer i solution.U:", solution.U)
    print("Truck arrival timestamp solution.T:", solution.T)
    print("Drone arrival timestamp solution.TDrone:", solution.TDrone)

    i = 0 
    solution.X_routes = [i]
    while i != data.customerNum+1: 
        for j in range(0,len(solution.X[i])): 
            if round(solution.X[i][j]) != 0:
                solution.X_routes.append(j)
                i = j
    solution.X_routes[-1] = 0
    solution.X_routesNum += 1
    print('Truck path solution.X_routes:', solution.X_routes)

    for i in range(0, data.nodeNum):
        for j in range(0, data.nodeNum):
            for k in range(0, data.nodeNum):
                if round(solution.Y[i][j][k]) != 0:
                    solution.Y_routes.append((i, j, k))
                    solution.Y_routesNum += 1
    print('Drone path solution.Y_routes:', solution.Y_routes)

    return solution

def plotSolution(data, solution):
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{data.DroneCustomerNum} Drone-eligible Customers in total {data.customerNum} Customers,single truck")
    plt.scatter(data.corX[0], data.corY[0], c='black', alpha=1, marker=',', linewidths=2, label='depot')
    plt.scatter(data.corX_Truck[1:], data.corY_Truck[1:], c='blue', alpha=1, marker='o', linewidths=1, label='customer_Truck') 
    plt.scatter(data.corX_Drone, data.corY_Drone, c='red', alpha=1, marker='o', linewidths=1, label='customer_Drone')
    # id
    for i in range(0,data.customerNum+1):
        x_ = data.corX[i]
        y_ = data.corY[i]
        label = data.N_lst[i]
        plt.text(x_, y_, str(label), family='serif', style='italic', fontsize=10, verticalalignment="bottom", ha='left', color='k')
    # truck path
    for i in range(0, len(solution.X_routes)-1):  # a -b
        a = solution.X_routes[i]
        b = solution.X_routes[i + 1]
        x = [data.corX[a], data.corX[b]]
        y = [data.corY[a], data.corY[b]]
        plt.plot(x, y, color='blue', linewidth=1, linestyle='-')
    # drone path
    for k in range(0, solution.Y_routesNum):
        for i in range(0, 2):  # sortie (i,j,k)
            a = solution.Y_routes[k][i]
            b = solution.Y_routes[k][i + 1]
            x = [data.corX[a], data.corX[b]]
            y = [data.corY[a], data.corY[b]]
            plt.plot(x, y, color='red', linewidth=1, linestyle=':')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()
    return 0

def printSolution(data,solution):
    print('_____________________________________________')
    for index, Drone_route in enumerate(solution.Y_routes):
        cost = 0
        for i in range(len(Drone_route) - 1):
            cost += Drone_factor * data.costMatrix[Drone_route[i]][Drone_route[i + 1]]
        print(f"Drone_Route-{index + 1} : {Drone_route} , time cost: {cost}")
    print('_____________________________________________')
    cost = 0
    for i in range(0, len(solution.X_routes)-1):
        cost += data.costMatrix[solution.X_routes[i]][solution.X_routes[i + 1]]
    print(f"Truck_Route: {solution.X_routes} , time cost: {cost}")


def modelingAndSolve(data):
    m = Model('FSTSP')

    m.setParam('MIPGap', 0.01)

    X = [[[] for _ in range(0, data.nodeNum)] for _ in range(0, data.nodeNum)]  # x_ij
    for i in data.N_out_lst:
        for j in data.N_in_lst:
            if i != j:
                X[i][j] = m.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}")

    Y = [[[[] for _ in range(0, data.nodeNum)] for _ in range(0,data.nodeNum)] for _ in range(0,data.nodeNum)]  # Y_ijk

    for i in data.N_out_lst:
        for j in data.C_lst:
            if i != j:
                for k in data.N_in_lst:
                    if (i,j,k) in data.P_lst:
                        Y[i][j][k] = m.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{j}_{k}")

    U = [[] for _ in range(0, data.nodeNum)]  # U_i

    for i in data.N_in_lst:
        U[i] = m.addVar(vtype=GRB.CONTINUOUS,lb=1.0, ub=data.customerNum+2, name=f"U_{i}")

    T = [[] for _ in range(0, data.nodeNum)]  # T_i

    for i in data.N_lst:
        T[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"T_{i}")

    TDrone = [[] for _ in range(0, data.nodeNum)]  # TDrone_i

    for i in data.N_lst:
        TDrone[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"TDrone_{i}")

    P = [[[] for _ in range(0, data.nodeNum)] for _ in range(0, data.nodeNum)]  # P_ij
    for i in data.N_out_lst:
        for j in data.C_lst:
            if j != i:
                P[i][j] = m.addVar(vtype=GRB.BINARY, name=f"P_{i}_{j}")

    m.update()
    # print(var_lst)

    obj = LinExpr(0) 
    obj.addTerms(1, T[data.customerNum+1])
    m.setObjective(obj, sense=GRB.MINIMIZE)

    # constraints:
    # 1
    num = 0
    for j in data.C_lst:
        expr = LinExpr(0)
        for i in data.N_out_lst:
            if i != j:
                expr.addTerms(1, X[i][j])
        for i in data.N_out_lst:
            if i != j:
                for k in data.N_in_lst:
                    if (i,j,k) in data.P_lst:
                        expr.addTerms(1, Y[i][j][k])
        num += 1
        m.addConstr(expr == 1, f'C1_{num}')
    # 2
    num = 0
    expr = LinExpr(0)
    for j in data.N_in_lst:
        expr.addTerms(1, X[0][j])
    num += 1
    m.addConstr(expr == 1, f'C2_{num}')
    # 3
    num = 0
    expr = LinExpr(0)
    for i in data.N_out_lst:
        expr.addTerms(1, X[i][data.customerNum+1])
    num += 1
    m.addConstr(expr == 1, f'C3_{num}')
    # 4.
    num = 0
    for j in data.C_lst:
        expr = LinExpr(0)
        for i in data.N_out_lst:
            if i != j:
                expr.addTerms(1,X[i][j])
        for k in data.N_in_lst:
            if k != j:
                expr.addTerms(-1,X[j][k])
        num += 1
        m.addConstr(expr == 0, f'C4_{num}')
    # 5.
    num = 0
    for i in data.C_lst:
        for j in data.N_in_lst:
            if i != j:
                expr = LinExpr(0)
                expr.addTerms(1,U[i])
                expr.addTerms(-1,U[j])
                expr.addTerms(data.customerNum+2, X[i][j])
                num += 1
                m.addConstr(expr <= data.customerNum+1, f'C5_{num}')
    # 6
    num = 0
    for i in data.N_out_lst:
        expr = LinExpr(0)
        for j in data.C_lst:
            if j != i:
                for k in data.N_in_lst:
                    if (i, j, k) in data.P_lst:
                        expr.addTerms(1, Y[i][j][k])
        num += 1
        m.addConstr(expr <= 1,f'C6_{num}')
    # 7
    num = 0
    for k in data.N_in_lst:
        expr = LinExpr(0)
        for i in data.N_out_lst:
            if i != k:
                for j in data.C_lst:
                    if (i,j,k) in data.P_lst:
                        expr.addTerms(1,Y[i][j][k])
        num += 1
        m.addConstr(expr <= 1, f"C7_{num}")
    # 8
    num = 0
    for i in data.C_lst:
        for j in data.C_lst:
            if j != i:
                for k in data.N_in_lst:
                    if (i,j,k) in data.P_lst:
                        expr = LinExpr(0)
                        for h in data.N_out_lst:
                            if h != i:
                                expr.addTerms(1, X[h][i])
                        for l in data.C_lst:
                            if l != k:
                                expr.addTerms(1, X[l][k])
                        expr.addTerms(-2,Y[i][j][k])
                        num += 1
                        m.addConstr(expr >= 0, f'C8_{num}')
    # 9.
    num = 0
    for j in data.C_lst:
        for k in data.N_in_lst:
            if (0,j,k) in data.P_lst:
                expr = LinExpr(0)
                for h in data.N_out_lst:
                    if h != k:
                        expr.addTerms(1,X[h][k])
                expr.addTerms(-1,Y[0][j][k])
                num += 1
                m.addConstr(expr >= 0, f'C9_{num}')
    # 10.
    num = 0
    for i in data.C_lst:
        for k in data.N_in_lst:
            if k != i:
                expr = LinExpr(0)
                expr.addTerms(1, U[i])
                expr.addTerms(-1, U[k])
                for j in data.C_lst:
                    if (i,j,k) in data.P_lst:
                        expr.addTerms(data.customerNum+2,Y[i][j][k])
                num += 1
                m.addConstr(expr <= data.customerNum+1, f'C10_{num}')
    # 11
    num = 0
    for i in data.C_lst:
        expr = LinExpr(0)
        expr.addTerms(1,T[i])
        expr.addTerms(-1,TDrone[i])
        for j in data.C_lst:
            if j != i:
                for k in data.N_in_lst:
                    if (i,j,k) in data.P_lst:
                        expr.addTerms(M, Y[i][j][k])
        num += 1
        m.addConstr(expr <= M, f'C11_{num}')
    # 12
    num = 0
    for i in data.C_lst:
        expr = LinExpr(0)
        expr.addTerms(1,TDrone[i])
        expr.addTerms(-1,T[i])
        for j in data.C_lst:
            if j != i:
                for k in data.N_in_lst:
                    if (i, j, k) in data.P_lst:
                        expr.addTerms(M, Y[i][j][k])
        num += 1
        m.addConstr(expr <= M, f'C12_{num}')
    # 13
    num = 0
    for k in data.N_in_lst:
        expr = LinExpr(0)
        expr.addTerms(1, T[k])
        expr.addTerms(-1, TDrone[k])
        for i in data.N_out_lst:
            if i != k:
                for j in data.C_lst:
                    if (i, j, k) in data.P_lst:
                        expr.addTerms(M, Y[i][j][k])
        num += 1
        m.addConstr(expr <= M, f'C13_{num}')
    # 14.
    num = 0
    for k in data.N_in_lst:
        expr = LinExpr(0)
        expr.addTerms(1, TDrone[k])
        expr.addTerms(-1, T[k])
        for i in data.N_out_lst:
            if i != k:
                for j in data.C_lst:
                    if (i, j, k) in data.P_lst:
                        expr.addTerms(M, Y[i][j][k])
        num += 1
        m.addConstr(expr <= M, f'C14_{num}')
    # 15.
    num = 0
    for h in data.N_out_lst:
        for k in data.N_in_lst:
            if h != k:
                expr = LinExpr(0)
                expr.addTerms(1, T[h])
                expr.addTerms(-1, T[k])
                for l in data.C_lst:
                    if l != k:
                        for m_ in data.N_in_lst:
                            if (k,l,m_) in data.P_lst:
                                expr.addTerms(SL, Y[k][l][m_])
                for i in data.N_out_lst:
                    if i != k :
                        for j in data.C_lst:
                            if (i,j,k) in data.P_lst:
                                expr.addTerms(SR,Y[i][j][k])
                expr.addTerms(M,X[h][k])
                num += 1
                m.addConstr(expr + data.costMatrix[h][k] <= M, f'C15_{num}')# 34
    # 16.
    num = 0
    for j in data.C_Drone_lst:
        for i in data.N_out_lst:
            if i != j:
                expr = LinExpr(0)
                expr.addTerms(1,TDrone[i])
                expr.addTerms(-1,TDrone[j])
                for k in data.N_in_lst:
                    if (i,j,k) in data.P_lst:
                        expr.addTerms(M,Y[i][j][k])
                num += 1
                m.addConstr(expr + Drone_factor * data.costMatrix[i][j] <= M, f'C16_{num}')
    # 17
    num = 0
    for j in data.C_Drone_lst:
        for k in data.N_in_lst:
            if k != j:
                expr = LinExpr(0)
                expr.addTerms(1,TDrone[j])
                expr.addTerms(-1, TDrone[k])
                for i in data.N_out_lst:
                    if (i,j,k) in data.P_lst:
                        expr.addTerms(M,Y[i][j][k])
                num += 1
                m.addConstr(expr + Drone_factor * data.costMatrix[j][k] + SR <= M, f'C17_{num}')
    # 18.
    num = 0
    for k in data.N_in_lst:
        for j in data.C_lst:
            if j != k:
                for i in data.N_out_lst:
                    if (i,j,k) in data.P_lst:
                        expr = LinExpr(0)
                        expr.addTerms(1,TDrone[k])
                        expr.addTerms(-1, TDrone[j])
                        expr.addTerms(M,Y[i][j][k])
                        num += 1
                        m.addConstr(expr + Drone_factor * data.costMatrix[i][j] <= Drone_endurance + M, f'C18_{num}')
    # 19
    num = 0
    for i in data.C_lst:
        for j in data.C_lst:
            if j != i:
                expr = LinExpr(0)
                expr.addTerms(1, U[i])
                expr.addTerms(-1, U[j])
                expr.addTerms(data.customerNum+2, P[i][j])
                num += 1
                m.addConstr(expr >= 1, f'C19_{num}')
    # 20.
    num = 0
    for i in data.C_lst:
        for j in data.C_lst:
            if j != i:
                expr = LinExpr(0)
                expr.addTerms(1, U[i])
                expr.addTerms(-1, U[j])
                expr.addTerms(data.customerNum + 2, P[i][j])
                num += 1
                m.addConstr(expr <= data.customerNum + 1, f'C20_{num}')
    # 21.
    num = 0
    for i in data.C_lst:
        for j in data.C_lst:
            if j != i:
                expr = LinExpr(0)
                expr.addTerms(1, P[i][j])
                expr.addTerms(1, P[j][i])
                num += 1
                m.addConstr(expr == 1, f'C21_{num}')
    # 22
    num = 0
    for i in data.N_out_lst:
        for k in data.N_in_lst:
            if i != k:
                for l in data.C_lst:
                    if (l != i) and (l != k):
                        expr = LinExpr(0)
                        expr.addTerms(1, TDrone[k])
                        expr.addTerms(-1, TDrone[l])
                        expr.addTerms(M, P[i][l])
                        for j in data.C_lst:
                            if ((i, j, k) in data.P_lst) and (j != l):
                                expr.addTerms(M, Y[i][j][k])
                        for m_ in data.C_lst:
                            if (m_ != i) and (m_ != k) and (m_ != l):
                                for n in data.N_in_lst:
                                    if ((l, m_, n) in data.P_lst) and (n != i) and (n != k):
                                        expr.addTerms(M, Y[l][m_][n])
                        num += 1
                        m.addConstr(expr <= 3*M, f'C22_{num}')
    # 23
    m.addConstr(T[0] == 0, f'C23')

    m.addConstr(TDrone[0] == 0, f'C24')

    num = 0
    for j in data.C_lst:
        num += 1
        m.addConstr(P[0][j] == 1, f'C25_{num}')

    start_time = time.time()
    m.optimize()
    m.write('FSTSP.lp')
    if m.status == GRB.OPTIMAL:
        print("-" * 20, "Solved successfully.", '-' * 20)
        print(f"Solution time: {time.time() - start_time} s")
        print(f"Objective funciton: {m.ObjVal}")
        solution = getSolution(data,m)
        plotSolution(data, solution)
        printSolution(data, solution)
    else:
        print("No solution.")
    return m


if __name__ =="__main__":
    data_path = r'/Users/jingyili/Desktop/FSTSP/data/C101network.txt'
    customerNum = 9 #here 9 is the largest number, otherwise it will exceed the size limit
    Drone_endurance = 20
    M = 100
    SR = 0.1
    SL = 0.1
    Drone_factor = 0.5 

    data = readData(data_path, customerNum, Drone_endurance)
    print("data.N_lst(N):", data.N_lst)
    print("data.N_out_lst(N0):", data.N_out_lst)
    print("data.N_in_lst(N+):",data.N_in_lst)
    print("data.C_lst(C):", data.C_lst)
    print("data.C_Drone_lst(C'):", data.C_Drone_lst)
    print("data.P_lst(P):", data.P_lst)

    print("-" * 20, "Problem Information", '-' * 20)
    print(f'Number of nodes: {data.nodeNum}')
    print(f'Number of customers: {data.customerNum}')
    print(f'Number of drone delivery available: {data.DroneCustomerNum}')

    modelingAndSolve(data)
