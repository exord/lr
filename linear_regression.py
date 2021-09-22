"""Module with functions to perform linear regression and inference."""
import numpy as np
from scipy import linalg
from scipy import stats as st

import matplotlib.pyplot as plt

from sklearn import linear_model as lm
from sklearn import preprocessing as pre
from sklearn import metrics
from sklearn import model_selection as ms


class BayesianLinearRegression(object):
    def __init__(self, ):
        pass


class LinearRegressor(object):

    def __init__(self, ):

        self.X_train = None
        return

    def fit(self, X, t, sample_weight=None):
        """
        Perform model fit.

        :param np.ndarray or pd.DataFrame X: design matrix
        with dimensions (nsamples x nfeatures).
        """

        self.X_train = X

    def inference(self):
        """Perform inference on model parameters."""
        if self.X_train is None:
            raise ValueError('Regressor not fit; use LinearRegressor.fit '
                             'first.')

        # res = self.X_train - self.predict(X_train)
        print('')
        return


def inference(params, intercept, X, res):
    """Perform inference on model parameters."""
    # Obtain estimate of variance
    ms_res = np.std(res, ddof=len(params)+1)  # Mean squared residuals2

    L = linalg.cho_factor(X.T @ X)
    var_estimate = ms_res * linalg.cho_solve(L, np.eye(X.shape[0]))

    for par in params:
        # Build statistics
        print(var_estimate)
    return


def hat_matrix(X, include_bias=True):
    """
    Compute hat matrix for design matrix X.

    :param np.array X: design matrix of dimensions (n x d),
    where n is the number of observations and d is the number of
    features.
    :param bool include_bias: if True (default), then include a bias column,
    in design matrix X (i.e. a column of ones - acts as an
    intercept term in a linear model).
    """
    if include_bias:
        X = np.hstack([np.ones([len(X), 1]), X])

    A = np.matmul(X.T, X)

    LL = linalg.cho_factor(A)
    return np.matmul(X, linalg.cho_solve(LL, X.T))


def vif(X, target_columns=None):
    """
    Compute the Variance Inflation Factor (VIF) for a given dataset.

    :param pd.DataFrame X: design matrix as a Pandas Data Frame (i.e. with
    column names, etc.)
    :param list target_column: columns to use as target. If None use all.

    :return dict outdict: a dictionary with the VIF for each feature.
    """
    if target_columns is None:
        target_columns = X.columns

    outdict = {}
    lrdict = {}
    for c in target_columns:
        Xi = X.copy()

        # Asssign label
        t_ = Xi.loc[:, c]
        # Drop target
        Xi.drop(c, axis='columns', inplace=True)

        assert c not in Xi.columns, 'No saqué la columna'

        # Scale features
        scaler = pre.StandardScaler()
        Xi_ = scaler.fit_transform(Xi)
        lr = lm.LinearRegression(fit_intercept=True)
        lr = lr.fit(Xi_, t_)

        outdict[c] = metrics.r2_score(t_, lr.predict(Xi_))
        lrdict[c] = [Xi.columns, lr.coef_]

    return outdict, lrdict


def analysis_regression(regressor, X_train, t_train, err_t_train=None,
                        param_names=None, loocv=True, plot=True,
                        include_bias=True,
                        plot_kwargs={}):
    """
    Perform inference and residual analyses for linear regresor.

    :param XX regressor: LinearRegressor, must implement methods predict and
    coef_
    :param X_train: design matrix with covariates
    :param t_train: target values.
    :param err_t_train: uncertainties on targets. If None, do OLS, otherwise
    perform WLS.
    """
    OLS = True if err_t_train is None else False

    # Compute model predictions and residuals
    y_train = regressor.predict(X_train)
    res = t_train - y_train

    # Compute weight matrix and variance-covariance matrix of coefficientes
    if OLS:
        W = np.eye(X_train.shape[0])
    else:
        W = np.diag(1/err_t_train**2)

    coefs = regressor.coef_

    print(X_train.shape, W.shape)
    L = linalg.cho_factor(X_train.T @ W @ X_train)
    V = linalg.cho_solve(L, np.eye(X_train.shape[1]))

    # Inference on parameters
    if OLS:
        # Compute estimate of data Variance
        ssres = np.sum((t_train - y_train)**2)
        dof = len(t_train) - len(coefs)
        msres = ssres / dof

        err_t_train = np.full_like(t_train, np.sqrt(msres))
        # Estimate of estimator variance
        cov_matrix = msres * V
        var_coefs = np.diag(cov_matrix)
        # Compute test statistics
        T = coefs/np.sqrt(var_coefs)
        # p-values are based on t-Student distribution
        tdist = st.t(df=dof)
        pvalues = 1 - tdist.cdf(np.abs(T))
    else:
        # In WLS, variance is provided
        cov_matrix = V.copy()
        var_coefs = np.diag(cov_matrix)
        T = coefs/np.sqrt(var_coefs)
        # p-values are based on normal distribution
        pvalues = 2 * (1 - st.norm.cdf(np.abs(T)))

    stars = np.where(pvalues > 0.05, '*', '')
    #
    if param_names is None:
        try:
            param_names = X_train.columns
        except AttributeError:
            param_names = range(1, X_train.shape[1]+1)

    # Print information
    print('{:<20}{:<22}  {:>7}  {:<3}'.format(
        'Parameter', 'Value', 'stat', 'p-value'))
    print('{:<20}{:<22}  {:>7}  {:<7}'.format(
        '=========', '=====', '====', '======='))
    for i, c in enumerate(coefs):
        print('{:<20}{: .2e} +/- {:.2e}  {:>7.2f}  {:.1e} {:1}'
              ''.format(param_names[i], c, np.sqrt(var_coefs)[i], T[i],
                        pvalues[i], stars[i]))

    # Print statistics
    if OLS:
        rmse = np.sqrt(metrics.mean_squared_error(t_train, y_train))
        R2 = metrics.r2_score(t_train, y_train)
        w = None
    else:
        w = 1/err_t_train**2
        rmse = np.sqrt(metrics.mean_squared_error(t_train, y_train,
                                                  sample_weight=w))
        R2 = metrics.r2_score(t_train, y_train, sample_weight=w)

    print('------------')
    dof = len(X_train)-len(coefs)
    if regressor.fit_intercept:
        dof -= 1
    print('Residual standard error: {:.2e} on {:d} dof'
          ''.format(res.std(ddof=len(coefs)), dof))
    print('Root mean squared error: {:.2e}'.format(rmse))
    print('R^2 score: {:.2f}'.format(R2))

    if loocv:
        loo = ms.LeaveOneOut()
        cv_score = ms.cross_val_score(regressor, X_train, t_train,
                                      scoring='neg_root_mean_squared_error',
                                      fit_params={'sample_weight': w},
                                      cv=loo)

        print('LOOCV Root mean squared error: {:.2e}+/-{:.2e}'.format(
            -cv_score.mean(), cv_score.std()))

    # Multiple R-squared:  0.6095,        Adjusted R-squared:  0.6055
    # F-statistic: 152.9 on 1 and 98 DF,  p-value: < 2.2e-16
    if plot:
        fig = plt.figure(figsize=(12, 12))

        # Predictions - Targets
        ax = fig.add_subplot(311)
        if OLS:
            ax.plot(y_train, t_train, 'o', **plot_kwargs)
        else:
            ax.errorbar(y_train, t_train, err_t_train,
                        fmt='o', **plot_kwargs)
        ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                '-k', lw=2)
        ax.set_xlabel('Model predictions', fontsize=16)
        ax.set_ylabel('Target values', fontsize=16)

        # Residuals - targets
        ax = fig.add_subplot(312)
        if OLS:
            ax.plot(y_train, res, 'o', **plot_kwargs)
        else:
            ax.errorbar(y_train, res, err_t_train,
                        fmt='o', **plot_kwargs)
        ax.axhline(0.0, ls='-', color='k', lw=2)
        ax.set_xlabel('Model predictions', fontsize=16)
        ax.set_ylabel('Residuals', fontsize=16)

        # Leverage
        hii = np.diag(hat_matrix(X_train, include_bias))
        ax = fig.add_subplot(313)
        resst = res/(err_t_train * np.sqrt(1 - hii))
        ax.plot(hii, resst, 'o', **plot_kwargs)
        ax.axhline(0.0, ls='-', color='k', lw=2)
        ax.set_xlabel('Leverage', fontsize=16)
        ax.set_ylabel('Standarised Residuals', fontsize=16)

    returndict = {'y': y_train, 'residuals': res, 'err': err_t_train,
                  'covariance': cov_matrix, 'leverage': hii}
    return returndict


def analysis_regression_gp(regressor, X_train, t_train, H_train,
                           param_names=None, loocv=True, plot=True,
                           plot_kwargs={}):
    """
    Perform inference and residual analyses for linear regresor.

    :param XX regressor: LinearRegressor, must implement methods predict and
    coef_
    :param X_train: design matrix with covariates
    :param t_train: target values.
    :param H_train: design matrix of linear model
    """
    # Compute model predictions and residuals
    y_train = regressor.full_predict(X_train, H_train)
    res = t_train - y_train

    # Inference on parameters
    coefs, cov_matrix = regressor.w_posterior_mean(
        return_cov=True, return_std=False)

    var_coefs = np.sqrt(np.diag(cov_matrix))
    T = coefs / var_coefs

    # p-values are based on t-Student distribution
    tdist = st.t(df=len(t_train) - len(coefs))
    pvalues = 1 - tdist.cdf(np.abs(T))

    stars = np.where(pvalues > 0.05, '*', '')
    #
    if param_names is None:
        try:
            param_names = X_train.columns
        except AttributeError:
            param_names = range(1, X_train.shape[1]+1)

    # Print information
    print('{:<20}{:<22}\t{: <10}\t{:<4}'.format(
        'Parameter', 'Value', 'Statistic', 'p-value'))
    print('{:<20}{:<22}\t{: <10}\t{:<7}'.format(
        '========', '=====', '=========', '======='))
    for i, c in enumerate(coefs):
        print('{:<20}{: .2e} +/- {:.2e}\t{: <10.2f}\t{:.2e} {:1}'
              ''.format(param_names[i], c, np.sqrt(var_coefs)[i], T[i],
                        pvalues[i], stars[i]))

    # Compute diagonal of kernel
    err2 = np.diag(regressor.kernel_(X_train.reshape(-1, 1)))
    # Print statistics
    w = 1/err2
    rmse = np.sqrt(metrics.mean_squared_error(t_train, y_train,
                                              sample_weight=w))

    print('------------')
    print('Residual standard error: {:.2e} on {:d} degrees of freedom'
          ''.format(res.std(ddof=len(coefs)), len(X_train)-len(coefs)))
    print('Root mean squared error: {:.2e}'.format(rmse))

    # if loocv:
    #     loo = ms.LeaveOneOut()
    #     cv_score = ms.cross_val_score(regressor, X_train, t_train,
    #                                   scoring='neg_root_mean_squared_error',
    #                                   fit_params={'sample_weight': w},
    #                                   cv=loo)
    #
    #     print('LOOCV Root mean squared error: {:.2e}+/-{:.2e}'.format(
    #         -cv_score.mean(), cv_score.std()))

    if plot:
        fig = plt.figure(figsize=(12, 12))

        # Predictions - Targets
        ax = fig.add_subplot(311)
        ax.plot(y_train, t_train, 'o', **plot_kwargs)
        ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                '-k', lw=2)
        ax.set_xlabel('Model predictions', fontsize=16)
        ax.set_ylabel('Target values', fontsize=16)

        # Residuals - targets
        ax = fig.add_subplot(312)
        ax.plot(y_train, res, 'o', **plot_kwargs)
        ax.axhline(0.0, ls='-', color='k', lw=2)
        ax.set_xlabel('Model predictions', fontsize=16)
        ax.set_ylabel('Residuals', fontsize=16)

        # Leverage
        hii = np.diag(hat_matrix(X_train, include_bias))
        ax = fig.add_subplot(313)
        resst = res/(np.sqrt(err2) * np.sqrt(1 - hii))
        ax.plot(hii, resst, 'o', **plot_kwargs)
        ax.axhline(0.0, ls='-', color='k', lw=2)
        ax.set_xlabel('Leverage', fontsize=16)
        ax.set_ylabel('Standarised Residuals', fontsize=16)

    returndict = {'y': y_train, 'residuals': res, 'err': np.sqrt(err2),
                  'covariance': cov_matrix, 'leverage': hii}
    return returndict


def anova(t, y_base, y_model, nparam_base, nparam_models,
          err_y=None, model_names=None):
    """
    Perform simple ANOVA analysis.

    :param np.array t: label array (dimensions (nsamples, 1) or (nsamples,))
    :param np.array y_base: predictions from base model
    (dimensions (nsamples, 1) or (nsamples,))
    :param np.array y_model: predictions from new (more complex) models
    (dimensions (nmodels, nsamples)
    :param int nparam_base: number of parameters of base model
    :param list nparam_models: list with number of parameters of new models
    :param list model_names: list with names of new models (optional)
    """
    y_model = np.atleast_2d(y_model)

    print('{:<10}\tdof\t{:<20}\tdof\tF-stat\tp-value'.format('Model',
                                                             'diferencia'))
    print('{:<10}\t---\t{:<20}\t---\t------\t-------'.format('-----',
                                                             '----------'))

    print('{:<10}\tN-{:d}'.format('Base', nparam_base))

    if err_y is None:
        err_y = y_base * 0.0 + 1.0

    for i, [y, npar] in enumerate(zip(y_model, nparam_models)):
        # Compute squared sums
        # Wikipedia https://en.wikipedia.org/wiki/F-test#Regression_problems
        # also Apunte by M.E. Szretter
        screg = np.sum((t - y_base)**2 / err_y**2)  # base model
        screg -= np.sum((t - y)**2 / err_y**2)  # new model
        screg /= (npar - nparam_base)  # normalise by difference in params

        scres = np.sum((t - y)**2 / err_y**2) / (len(t) - npar)

        fratio = screg/scres

        # Define appropiate F distribution
        my_f = st.f(dfn=(npar - nparam_base), dfd=(len(t) - npar))
        pvalue = 1 - my_f.cdf(fratio)

        if model_names is not None:
            name = model_names[i]
        else:
            name = 'New_{:d}'.format(i+1)

        printdict = {'model': name,
                     'dif': name+' - Base',
                     'npar': npar,
                     'dpar': npar - nparam_base,
                     'fratio': fratio,
                     'pvalue': pvalue
                     }
        # Print line in table
        print('{model:<10}\tN-{npar:d}\t{dif:<20}\t{dpar:d} '
              '\t{fratio:.4f}\t{pvalue:.2e}'.format(**printdict))

    return
