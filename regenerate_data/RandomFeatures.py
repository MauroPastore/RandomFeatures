#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:46:36 2022

@author: fabian
"""
from jax import nn
from jax import jit
# from jax import grad
from jax import random
import jax.numpy as jnp
# import jax.lax as lax
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


from datetime import datetime



from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures



def perceptron_fun(theta,x):
    return jnp.sign(jnp.dot(theta,x))

def twoLayer(X,F,theta):
    return jnp.sign(jnp.dot(theta,jnp.sign(jnp.dot(F,X))))

def higher_perceptron(X, theta1, theta2, gamma):

    D,P  = X.shape
    theta1 = theta1.reshape(D,)

    # r = jnp.dot(theta1,X)/jnp.sqrt(D)  + jnp.diag(jnp.dot(X.T,jnp.dot(theta2,X)))/D
    r = gamma * jnp.dot(theta1,X)/jnp.sqrt(D)  + jnp.sqrt(1 - gamma**2)* jnp.diag(jnp.dot(X.T,jnp.dot(theta2,X)))/D/2
    return r



def trainerrorRF(params, data, perceptron):
    if perceptron:
        return trainerrorRF_perceptron(params, data, perceptron)
    else:
        return trainerrorRF_continuous(params, data, perceptron)

@jit
def trainerrorRF_perceptron(params, data, perceptron):
    w = params
    X, y, D, P, N,_,F,_ = data
    ynew = jnp.dot(w,nn.elu(jnp.dot(F,X)/jnp.sqrt(D)))/jnp.sqrt(N)
    ynew = jnp.sign(ynew)
    return jnp.mean((y-ynew)**2)/4

@jit
def trainerrorRF_continuous(params, data, perceptron):
    w = params
    X, y, D, P, N,_,F,beta= data
    ynew = jnp.dot(w,nn.elu(jnp.dot(F,X)/jnp.sqrt(D)))/jnp.sqrt(N)
    ynew = jnp.tanh(beta*ynew)*(1 + 1/beta)
    return jnp.mean((y-ynew)**2)


def trainerrorLin(params, data, perceptron ):
    w = params
    X, y, D, P, _,_,_ ,beta= data

    if len(w) ==X.shape[0]:
        ynew = jnp.dot(w,X)


    else:
        Xq = poly.fit_transform(X.T)
        ynew = jnp.dot(w,Xq.T)
    if perceptron:
        ynew = jnp.sign(ynew)
        return jnp.mean((y-ynew)**2)/4
    else:
        ynew = jnp.tanh(beta*ynew)*(1 +1/beta)
        return jnp.mean((y-ynew)**2)

def errorstats(elist):
    return (jnp.mean(jnp.array(elist)),jnp.sqrt(jnp.var(jnp.array(elist))))

def quad_features(X):
    Xq = X
    return Xq




def produce_plots(VV,label):
    plt.errorbar(VV/D,jnp.array(trainlist)[:,0],yerr = jnp.array(trainlist)[:,1], label = 'RF - train')
    plt.errorbar(VV/D,jnp.array(genlist)[:,0],yerr = jnp.array(genlist)[:,1], label = 'RF - test')
    # plt.plot(VV/D,jnp.log10(VV)*0 + jnp.mean(jnp.array(glinearlist)[:,0]),'--', label = 'linear')
    # plt.plot(VV/D,jnp.log10(VV)*0 + jnp.mean(jnp.array(gquadlist)[:,0]), label = 'quad')
    plt.plot(VV/D,m1list,':o', label = 'm1')
    plt.plot(VV/D,m2list, ':*', label = 'm2')
    plt.plot(VV/D,VV*0 + ynu*gamma/mu1)
    plt.plot(VV/D,VV*0 + jnp.sqrt(2)*ynu*np.sqrt(1-gamma**2)/mu2)
    plt.xscale("log")
    plt.xlabel(label)
    # if gamma ==1:
    #     plt.title(f"D = {D}, P={P},  g = 1, perceptron = {perceptron}, eps = {eps}")
    # else:
    #     plt.title(f"D = {D}, P={P},  g = {gamma**2:0.1}, perceptron = {perceptron}, eps = {eps}")
    plt.legend()


# @jit
def overlaps(vals):
    w,F,theta,G = vals
    N,D =  F.shape
    s1 = jnp.dot(w,F)/jnp.sqrt(N)
    m1 = jnp.dot(s1,theta)/D
    m2 = 0

    m2 = jnp.dot(w,jnp.einsum('ij,ji->i',jnp.dot(F,G),F.T))

    m2 = m2/jnp.sqrt(N)/D**2

    return m1,m2


def savedata():
    df = pd.DataFrame(genlist, columns = ["gen","gensd"])
    df["train"] = pd.DataFrame(trainlist, columns = ["train","trainsd"])["train"]
    df["trainsd"] = pd.DataFrame(trainlist, columns = ["train","trainsd"])["trainsd"]
    df["m1"] = pd.DataFrame(m1list, columns = ["m1","m1sd"])["m1"]
    df["m1sd"] = pd.DataFrame(m1list, columns = ["m1","m1sd"])["m1sd"]
    df["m2"] = pd.DataFrame(m2list, columns = ["m2","m2sd"])["m2"]
    df["m2sd"] = pd.DataFrame(m2list, columns = ["m2","m2sd"])["m2sd"]
    df["N"] = N*np.ones(shape = (ts + 1,), dtype = int)
    df["D"] = D*np.ones(shape = (ts + 1,), dtype = int)
    df["P"] = PP[0:ts + 1]
    df["eps"] = eps*np.ones(shape = (ts + 1,))
    df["gamma"] = gamma*np.ones(shape = (ts + 1,))
    df["T"] = T*np.ones(shape = (ts + 1,), dtype = int)
    df["zeta"] = lamw*np.ones(shape = (ts + 1,))
    df["solver"] = solverlist
    df["lingen"] = pd.DataFrame(glinearlist, columns = ["lingen","lingensd"])["lingen"]
    df["lingensd"] = pd.DataFrame(glinearlist, columns = ["lingen","lingensd"])["lingensd"]
    df["quadgen"] = pd.DataFrame(gquadlist, columns = ["quadgen","quadgensd"])["quadgen"]
    df["quadgensd"] = pd.DataFrame(gquadlist, columns = ["quadgen","quadgensd"])["quadgensd"]
    return df




if __name__ == '__main__':


    perceptron = True


    NAME = "test.csv"
    mode = "w"


    lamw = 0.0001


    tic = time.perf_counter()
    D = 100
    P = int(100)
    N = 200
    seed = int(time.time())
    key = random.PRNGKey(seed)
    ratew = 0.0001
    rate = 0.1
    T = 20

    beta = 1e-5

    # gamma = jnp.sqrt(0.5)
    gamma = 1

    eps = 0

    # NN = jnp.logspace(2,3,num =5,dtype=int)

    PP = jnp.logspace(1,2,num = 5, dtype = int)

    # NN = jnp.linspace(100,1000,num =25,dtype=int)


    trainlist = list()
    genlist = list()
    glinearlist = list()
    gquadlist = list()

    poly = PolynomialFeatures(2, include_bias = False, interaction_only = True)


    m1list = list()
    m2list = list()




    mu0 = 1/np.sqrt(2*np.pi)
    mu1 = 1/2
    mu2 = mu0
    mus = np.sqrt(1/2 - 1/2/np.pi - 1/4 - mu2**2/2)

    muvec = mu0,mu1,mu2,mus
    ynu = np.sqrt(2/np.pi)

    gtlist = list()
    solverlist = list()
    solver = 'auto'

    for ts,P in enumerate(PP):
        terror = list()
        gerror = list()
        glinearerror = list()
        gquaderror = list()

        v = 0
        v2 = 0
        data = D,N,P,muvec,lamw,rate


        m1s = list()
        m2s = list()
        solverlist.append(solver)

        for t in range(T):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f"Doing sample {t} out of {T} of P={P} point {ts} out of {len(PP)} at {current_time}\n")
            key, F_key, w_key, theta_key, G_key, data_key = random.split(key, 6)


            F = random.normal(F_key, shape = (N,D))
            w = random.normal(theta_key, shape=(1,N))
            X = random.normal(data_key, shape = (D,P))


            theta = random.normal(theta_key, shape=(D,))
            G = random.normal(G_key, shape = (D,D))
            G = G/jnp.sqrt(2) + G.T/jnp.sqrt(2)
            G = G - jnp.diag(G)

            key, noise_key = random.split(key)

            if perceptron:
                y = jnp.sign(higher_perceptron(X, theta, G, gamma))
            else:
                y = higher_perceptron(X, theta, G, gamma) +  eps*random.normal(noise_key, shape = (P,))
                y = jnp.tanh(beta*y)*(1+1/beta)

            params = w
            data = X, y, D, P, N, lamw, F,beta


            info = params,data,ratew

            clf = Ridge(random_state=0, fit_intercept=False,  alpha = lamw, max_iter = 15000, solver =solver).fit(nn.elu(jnp.dot(F,X)/jnp.sqrt(D)).T/jnp.sqrt(N), y)



            Xq = poly.fit_transform(X.T)
            clf2  = Ridge(fit_intercept=False, alpha = lamw).fit(Xq,y)

            clflin = Ridge(fit_intercept=False, alpha = lamw).fit(X.T,y)

            w = clf.coef_

            thetah = clf2.coef_
            thetalin = clflin.coef_


            m1 , m2 = overlaps((w, F, theta, G))
            m1s.append(m1)
            m2s.append(m2)
            v = v + m1/T
            v2 =  v2 + m2/T

            key, data_key = random.split(key,2)
            Xtest = random.normal(data_key, shape = (D,P))

            if perceptron:
                ytest = jnp.sign(higher_perceptron(Xtest, theta, G, gamma))
            else:
                ytest = higher_perceptron(Xtest, theta, G, gamma)
                ytest = jnp.tanh(beta*ytest)*(1+1/beta)
            datatest = Xtest, ytest, D, 2*P, N, lamw, F,beta


            params = w
            info = params,data,ratew
            terror.append(trainerrorRF(params, data, perceptron = perceptron))
            gerror.append(trainerrorRF(params, datatest, perceptron = perceptron))
            glinearerror.append(trainerrorLin(thetalin,datatest, perceptron = perceptron))
            gquaderror.append(trainerrorLin(thetah,datatest, perceptron = perceptron))

        m1list.append(errorstats(m1s))
        m2list.append(errorstats(m2s))
        trainlist.append(errorstats(terror))
        genlist.append(errorstats(gerror))
        glinearlist.append(errorstats(glinearerror))
        gquadlist.append(errorstats(gquaderror))
        df = savedata()
        df.to_csv(NAME,index = False,mode = mode)



    # produce_plots(PP, "P/D")
    df = savedata()
    df.to_csv(NAME,index = False, mode = mode)





    toc = time.perf_counter()

    print(toc-tic)
