import numpy as np
import scipy
import scipy.stats

def KGE2012_ALL(obs,model):
  #Use only non-nan values
  idx = np.where((np.isnan(obs) == 0) & (np.isnan(model) == 0))
  rho = scipy.stats.pearsonr(model[idx],obs[idx])[0]
  if np.isnan(rho) == 1:rho = 0.0
  mean_ratio = np.mean(model[idx])/np.mean(obs[idx])
  cv_ratio = np.std(model[idx])/np.std(obs[idx])/mean_ratio
  kge = 1 - ((rho - 1)**2 + (mean_ratio - 1)**2 + (cv_ratio - 1)**2)**0.5
  values = {'kge':kge,'rho':rho,'beta':mean_ratio,'alpha':cv_ratio}
  return values

def KGE2012(obs,model):
  #Use only non-nan values
  idx = np.where((np.isnan(obs) == 0) & (np.isnan(model) == 0))
  rho = scipy.stats.pearsonr(model[idx],obs[idx])
  mean_ratio = np.mean(model[idx])/np.mean(obs[idx])
  cv_ratio = np.std(model[idx])/np.std(obs[idx])/mean_ratio
  kge = 1 - ((rho[0] - 1)**2 + (mean_ratio - 1)**2 + (cv_ratio - 1)**2)**0.5
  return kge

def KGE(obs,model):
  #Use only non-nan values
  idx = np.where((np.isnan(obs) == 0) & (np.isnan(model) == 0))
  rho = scipy.stats.pearsonr(model[idx],obs[idx])
  mean_ratio = np.mean(model[idx])/np.mean(obs[idx])
  std_ratio = np.std(model[idx])/np.std(obs[idx])
  kge = 1 - ((rho[0] - 1)**2 + (mean_ratio - 1)**2 + (std_ratio - 1)**2)**0.5
  return kge

def KGE_ALL(obs,model):
  #Use only non-nan values
  idx = np.where((np.isnan(obs) == 0) & (np.isnan(model) == 0))
  rho = scipy.stats.pearsonr(model[idx],obs[idx])
  mean_ratio = np.mean(model[idx])/np.mean(obs[idx])
  std_ratio = np.std(model[idx])/np.std(obs[idx])
  kge = 1 - ((rho[0] - 1)**2 + (mean_ratio - 1)**2 + (std_ratio - 1)**2)**0.5
  values = {'kge':kge,'rho':rho[0],'beta':mean_ratio,'alpha':std_ratio}
  return values

def NSE(obs,model):
  #Use only non-nan values
  idx = np.where((np.isnan(obs) == 0) & (np.isnan(model) == 0))
  nse = 1 - np.sum((obs[idx] - model[idx])**2)/np.sum((obs[idx] - np.mean(obs[idx]))**2)
  return nse

def NSE_ALL(obs,model):
  #Use only non-nan values
  idx = np.where((np.isnan(obs) == 0) & (np.isnan(model) == 0))
  alpha = np.std(model[idx])/np.std(obs[idx])
  beta = (np.mean(model[idx]) - np.mean(obs[idx]))/np.std(obs[idx])
  rho = scipy.stats.pearsonr(model[idx],obs[idx])[0]
  nse = 2*alpha*rho - alpha**2 - beta**2
  mus = np.mean(model[idx])
  muo = np.mean(obs[idx])
  sto = np.std(model[idx])
  values = {'nse':nse,'rho':rho,'beta':beta,'alpha':alpha,'mus':mus,'muo':muo,'sto':sto}
  return values

def R2(obs,model):
  #Use only non-nan values
  idx = np.where((np.isnan(obs) == 0) & (np.isnan(model) == 0))
  rho = scipy.stats.pearsonr(model[idx],obs[idx])
  return rho[0]**2

def R(obs,model):
  #Use only non-nan values
  idx = np.where((np.isnan(obs) == 0) & (np.isnan(model) == 0))
  rho = scipy.stats.pearsonr(model[idx],obs[idx])
  return rho[0]

def nRMSE(obs,model):
  #Use only non-nan values
  idx = np.where((np.isnan(obs) == 0) & (np.isnan(model) == 0))
  if len(idx[0]) > 1:
   rmsd = (np.mean((model[idx] - obs[idx])**2))**0.5
   nrmsd = 100*rmsd/(np.max(obs[idx]) - np.min(obs[idx]))
  else:
   nrmsd = float('NaN')
  return nrmsd

def RMSE(obs,model):
  #Use only non-nan values
  idx = np.where((np.isnan(obs) == 0) & (np.isnan(model) == 0))
  if len(idx[0]) > 1:
   rmse = (np.mean((model[idx] - obs[idx])**2))**0.5
  else:
   rmse = float('NaN')
  return rmse

def MAE(obs,model):
  #Use only non-nan values
  idx = np.where((np.isnan(obs) == 0) & (np.isnan(model) == 0))
  if len(idx[0]) > 1:
   abias = np.mean(np.abs(model[idx] - obs[idx]))
  else:
   abias= np.nan
  return abias

