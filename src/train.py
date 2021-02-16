
def main(params, data):
    """
    INPUT:

    regressionalgs (dict):
    datasetName(str):
    trainsize (float):
    testsize (float):
    numruns (int):
    parameters (dict):
    """
    for alg in params["algorithms"]:
        if alg[-1] == 'R':
            mod = importlib.import_module("algorithms.regressors")
            class_ = getattr(mod, alg)

    numalgs = len(regressionalgs)
    numparams = len(parameters)
    errors = {}
    errorstrain = {}
    index_split = int((data.shape[0])*(params["train/test"][0]/100))
    X_t=data[:data.shape[0]+1,:data.shape[1]-1]
    Y_t=data[:data.shape[0]+1,data.shape[1]-1:]

    X_split = np.split(X_t,[index_split])
    Y_split = np.split(Y_t,[index_split])

    X_t_train = X_split[0]
    y_t_train = Y_split[0]
    X_t_test = X_split[1]
    y_t_test = Y_split[1]
    #Creates error arrays for each regressor.
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))
        errorstrain[learnername] = np.zeros((numparams,numruns))
    for r in range(numruns):
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))
        for p in range(numparams):
            params = parameters[p]
            print (params)
            for learnername, learner in regressionalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model on (test data)
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error
                # Test model on (train data)
                predictions = learner.predict(trainset[0])
                errort = geterror(trainset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errorstrain[learnername][p,r] = errort

    #Obtain best algorithm with the best hyperparameter setting
    for learnername in regressionalgs:
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        besterrortrain = np.mean(errorstrain[learnername][0,:])
        bestparams = 0

        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            standerror= np.std(errors[learnername][p,:],ddof=1)/math.sqrt(numruns) # Compute the standar deviation with n-1
            aveerrortrain = np.mean(errorstrain[learnername][p,:])
            standerrortrain= np.std(errorstrain[learnername][p,:],ddof=1)/math.sqrt(numruns) # Compute the standar deviation with n-1
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p
            if aveerrortrain < besterrortrain:
                besterrortrain = aveerrortrain
                bestparams = p

        # Extract best parameters
        #learner.reset(parameters[bestparams])
        print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average test error for ' + learnername + ': ' + str(besterror))
        print ('Standard test error for ' + learnername + ': ' + str(standerror))# calcultae the sample standar deviation for the test error
        print ('Average train error for ' + learnername + ': ' + str(besterrortrain))
        print ('Standard train error for ' + learnername + ': ' + str(standerrortrain))# calcultae the sample standar deviation for the train error
