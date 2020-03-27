########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        # unfortunately, numpy is the only data structure I know
        # besides, it can do matrix multiplication :)
        A = np.array(self.A).T
        O = np.array(self.O).T
        A_start = np.array(self.A_start)

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        # probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        # seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]
        probs = np.zeros((self.L, M+1), dtype=np.float)
        seqs = np.zeros((self.L, M+1), dtype='|U256')

        ### Use standard notation, our matrice are all transposed
        ### A(i,j) is the prob from j to i,  L by L
        ### O(i,j) is the prob from j to i,  D by L
        ### TODO: Insert Your Code Here (2A)

        probs[:,0] = A_start
        probs[:,1] = O[x[0]] * probs[:,0]

        for i in range(self.L):
            seqs[i, 0] = str(i)
            seqs[i, 1] = str(i)

        for t in range(2,M+1):
            obs = x[t-1]
            for i in range(self.L): # for current Y state
                P = np.zeros((self.L,))
                for j in range(self.L): # for previous Y state
                    Pprev = probs[j,t-1]
                    Ptrans = A[i,j]
                    Pobs = O[obs, i]
                    P[j] = Pprev * Ptrans * Pobs
                probs[i,t] = np.max(P)
                seqs[i,t] = seqs[np.argmax(P), t-1] + str(i)

        best = np.argmax(probs[:, M])
        max_seq = seqs[best, M]
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        A = np.array(self.A).T
        O = np.array(self.O).T
        A_start = np.array(self.A_start)
        # alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        alphas = np.zeros((self.L, M+1))

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2Bi)
        alphas[:,0] = A_start
        alphas[:,1] = O[x[0]] * alphas[:,0]

        for t in range(2, M + 1):
            obs = x[t - 1]
            alphas[:,t] = O[obs, :] * np.matmul(A,  alphas[:,t-1])

            if normalize:
                alphas[:,t] = alphas[:,t]/np.sum(alphas[:,t])

        return alphas.T


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        A = np.array(self.A).T
        O = np.array(self.O).T
        A_start = np.array(self.A_start)
        # betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        betas = np.zeros((self.L, M+1))

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2Bii)
        betas[:,M] = 1

        for t in reversed(range(0, M)):
            obs = x[t]
            betas[:,t] = np.matmul(A.T,  (betas[:,t+1]*O[obs, :]))

            if normalize:
                betas[:,t] =betas[:,t]/np.sum(betas[:,t])

        return betas.T


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        dataset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        A = np.array(self.A).T
        O = np.array(self.O).T

        N = len(X)


        # Calculate each element of A using the M-step formulas.

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2C)
        Anumerator = np.zeros(A.shape)
        Adenominator = np.zeros(A.shape)
        Onumerator = np.zeros(O.shape)
        Odenominator = np.zeros(O.shape)

        for j in range(N):
            M = len(X[j])
            for t in range(M):
                y = Y[j][t]
                x = X[j][t]
                if t != M-1:
                    y_next =Y[j][t+1]

                    Anumerator[y_next,y] += 1
                    Adenominator[:, y] += 1

                Onumerator[x, y] += 1
                Odenominator[:, y] += 1

        Adenominator[Adenominator == 0] = 1
        Odenominator[Odenominator == 0] = 1

        A = Anumerator / Adenominator
        O = Onumerator / Odenominator
        self.A = A.T.tolist()
        self.O = O.T.tolist()

        return


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2D)

        N = len(X)


        for iter in range(N_iters):
            print('iter:', iter)
            A = np.array(self.A).T
            O = np.array(self.O).T
            Anumerator = np.zeros(A.shape)
            Adenominator = np.zeros(A.shape)
            Onumerator = np.zeros(O.shape)
            Odenominator = np.zeros(O.shape)
            for j in range(N):
                M = len(X[j])

                alphas = self.forward(X[j], normalize=True).T
                betas = self.backward(X[j], normalize=True).T
                for t in range(M):
                    x = X[j][t]
                    alpha_last = alphas[:, t]
                    alpha = alphas[:,t+1]
                    beta = betas[:,t+1]

                    Pxy = (alpha * beta) / np.dot(alpha, beta)
                    Onumerator[x, :] += Pxy
                    Odenominator[:, :] += np.tile(Pxy.reshape(1, self.L), [self.D, 1])

                    if t != 0:
                        Pba = np.tile(alpha_last.reshape(1,self.L),[self.L,1]) \
                              * A \
                              * np.tile(beta.reshape(self.L,1),[1,self.L]) \
                              * np.tile(O[x].reshape(self.L,1),[1,self.L])
                        Anumerator[:, :] += Pba / np.sum(Pba,axis=(0,1))
                    if t!= M-1:
                        Adenominator[:, :] += np.tile(Pxy.reshape(1, self.L), [self.L, 1])


            # print(Anumerator)
            # print(Adenominator)
            # print(Onumerator)
            # print(Odenominator)
            Adenominator[Adenominator == 0] = 1
            Odenominator[Odenominator == 0] = 1

            A = Anumerator / Adenominator
            O = Onumerator / Odenominator

            self.A = A.T.tolist()
            self.O = O.T.tolist()


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = np.zeros((M,),dtype=np.int)
        states = np.zeros((M,),dtype=np.int)

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2F)
        A = np.array(self.A).T
        O = np.array(self.O).T

        y0 = np.random.choice(self.L)
        states[0] = y0

        yt = y0
        for t in range(1,M):
            yt = np.random.choice(self.L,p=A[:,yt])
            states[t] = yt

        for t in range(M):
            emission[t] =  np.random.choice(self.D,p=O[:,states[t]])

        emission = emission.tolist()
        states = states.tolist()
        return emission, states

    def generate_emission2(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission1 = np.zeros((M,),dtype=np.int)
        emission2 = np.zeros((M,), dtype=np.int)
        states = np.zeros((M,),dtype=np.int)

        ###
        ###
        ###
        ### TODO: Insert Your Code Here (2F)
        A = np.array(self.A).T
        O = np.array(self.O).T

        y0 = np.random.choice(self.L)
        states[0] = y0

        yt = y0
        for t in range(1,M):
            yt = np.random.choice(self.L,p=A[:,yt])
            states[t] = yt

        for t in range(M):
            emission1[t] =  np.random.choice(self.D,p=O[:,states[t]])
            emission2[t] = np.random.choice(self.D, p=O[:, states[t]])

        emission1 = emission1.tolist()
        emission2 = emission2.tolist()
        states = states.tolist()
        return emission1, emission2, states

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations.update(set(x))
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    # random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    # random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
